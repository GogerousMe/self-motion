"""Streaming port of the user's frame-based MotionEnergy V1.

No CARLsim dependency. Arrays are S,F,Y,X (3,28,H,W), before spike sampling.
The source-compatible profile intentionally keeps scale>4 and RGB red=0.1299.
Derived constants/algorithm: UCI-CARL MotionEnergy; see THIRD_PARTY_NOTICES.md.
"""
from dataclasses import asdict, dataclass
from contextlib import contextmanager
from math import factorial, isfinite

import numpy as np

SOURCE_SHA256 = 'd869f55639618c39c41d9e7c7b980fb0701b247975ecad48484d2beef0eec488'
ORIENTATIONS = (
    (0.7246,-0.9718,0.7496,-0.5837,-0.0810,0.9439,0.3203,-0.8712,-0.1593,-0.5142,0.9304,0.3737,-0.8031,-0.8126,0.6004,-0.5738,0.0024,0.5969,0.1436,0.7757,-0.4004,-0.5108,0.2375,-0.2221,-0.5140,0.5194,-0.0870,0.3838),
    (-0.6559,-0.1019,0.6240,-0.7797,0.9692,-0.2312,-0.9151,0.4207,-0.9533,0.8175,0.2398,0.8810,-0.4430,0.0588,-0.5384,0.5644,0.7931,0.5142,-0.7680,-0.0669,-0.6670,-0.2747,0.5034,0.5042,0.1580,0.1332,-0.5159,-0.3549),
    (0.2113,0.2126,0.2210,0.2266,0.2327,0.2359,0.2451,0.2529,0.2567,0.2593,0.2772,0.2902,0.3984,0.5799,0.5913,0.5935,0.6091,0.6160,0.6241,0.6275,0.6283,0.8146,0.8308,0.8345,0.8431,0.8441,0.8522,0.8525),
)
GAUSS = (0.0007,0.0155,0.0903,0.2345,0.3179,0.2345,0.0903,0.0155,0.0007)
NORM = (0.0045,0.0072,0.0109,0.0160,0.0225,0.0303,0.0393,0.0490,0.0587,0.0675,0.0746,0.0792,0.0808,0.0792,0.0746,0.0675,0.0587,0.0490,0.0393,0.0303,0.0225,0.0160,0.0109,0.0072,0.0045)
DIFF = {1: (-0.5,0,0.5), 2: (1,-2,1), 3: (-0.5,1,0,-1,0.5)}


@dataclass(frozen=True)
class V1Config:
    profile: str = 'motion-energy-frame-20261028'
    linear_gain: float = 6.6084
    rect_gain: float = 1.9263
    complex_gain: float = 25.0
    norm_strength: float = 0.98
    c50: float = 0.1

    def __post_init__(self):
        if self.profile != 'motion-energy-frame-20261028':
            raise ValueError('unsupported V1 profile')
        if any(not isfinite(v) or v <= 0 for v in
               (self.linear_gain,self.rect_gain,self.complex_gain,self.norm_strength,self.c50)):
            raise ValueError('positive finite V1 gains required')


class ArrayOps:
    """Small device adapter; NumPy reference and optional torch CPU/CUDA."""
    def __init__(self, backend='numpy', device='cpu'):
        if backend not in ('numpy', 'torch'):
            raise ValueError('backend must be numpy or torch')
        if backend == 'numpy' and device != 'cpu':
            raise ValueError('NumPy requires device=cpu')
        self.backend, self.device, self.torch = backend, device, None
        self._validation_conditions = None
        if backend == 'torch':
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError('GPU execution requires PyTorch; install it or use --device cpu') from exc
            target = torch.device(device)
            if target.type not in ('cpu','cuda'):
                raise ValueError('only torch CPU/CUDA are supported')
            if target.type == 'cuda' and not torch.cuda.is_available():
                raise RuntimeError('CUDA requested but unavailable; check CUDA PyTorch/driver or use --device cpu; no silent CPU fallback')
            self.torch = torch
            self.device = str(target)
            # Reference validation must not use reduced-precision convolution.
            if target.type == 'cuda':
                torch.backends.cudnn.allow_tf32 = False
                torch.backends.cuda.matmul.allow_tf32 = False

    def array(self, value, double=False):
        if self.torch:
            return self.torch.as_tensor(value, dtype=self.torch.float64 if double else self.torch.float32,
                                        device=self.device)
        return np.asarray(value, dtype=np.float64 if double else np.float32)

    def zeros(self, shape):
        return self.torch.zeros(shape, dtype=self.torch.float32, device=self.device) if self.torch else np.zeros(shape,np.float32)

    def host(self, value):
        return value.detach().cpu().numpy() if self.torch else np.asarray(value)

    def float32(self, value):
        return value.float() if self.torch else value.astype(np.float32)

    def float64(self, value):
        return value.double() if self.torch else value.astype(np.float64)

    def stack(self, values):
        return self.torch.stack(values) if self.torch else np.stack(values)

    def copy(self, value):
        return value.clone() if self.torch else value.copy()

    def sync(self):
        if self.torch and self.device.startswith('cuda'):
            self.torch.cuda.synchronize(self.device)

    def check(self,condition):
        if self._validation_conditions is not None:
            self._validation_conditions.append(condition)
            return True
        return bool(condition)

    @contextmanager
    def validation_scope(self):
        """Collect device-side numerical checks; synchronize once at scope end.

        Keep pre-reset overflow flags, even when a later reset makes v finite.
        A failed frame must be discarded/reset; execution is not rolled back.
        """
        if self._validation_conditions is not None:raise RuntimeError('nested validation scopes unsupported')
        conditions=[];self._validation_conditions=conditions
        try:
            yield
            if conditions:
                passed=self.torch.stack(conditions).all() if self.torch else np.asarray(conditions).all()
                if not bool(passed):raise FloatingPointError('neural numerical validation failed during frame')
        finally:self._validation_conditions=None

    def correlate(self, value, kernel, axis):
        """Source kernels do correlation, not flipped convolution; zero padding."""
        radius = len(kernel)//2
        if self.torch:
            import torch.nn.functional as F
            moved = value.movedim(axis, -1)
            shape = moved.shape
            weights = self.torch.as_tensor(kernel, dtype=value.dtype, device=self.device).reshape(1,1,-1)
            result = F.conv1d(moved.reshape(-1,1,shape[-1]),weights,padding=radius)
            return result.reshape(shape).movedim(-1,axis)
        pad = [(0,0)]*value.ndim
        pad[axis] = (radius,radius)
        padded = np.pad(value,pad)
        result = np.zeros_like(value)
        for k, weight in enumerate(np.asarray(kernel,dtype=value.dtype)):
            sl = [slice(None)]*value.ndim
            sl[axis] = slice(k,k+value.shape[axis])
            result += padded[tuple(sl)]*weight
        return result


def edge_maps(width, height, dtype=np.float32):
    y,x = np.indices((height,width))
    distance = np.minimum(np.minimum(x,width-1-x),np.minimum(y,height-1-y))
    maps = [np.minimum(distance.astype(np.float64)**power,den)/den
            for power,den in ((3,125),(4,1296),(5,7776))]
    return np.asarray(maps,dtype=dtype)


class MotionEnergyV1:
    """Fixed frame cadence, zero initial history; reset only at sequence boundaries.

    step() returns a device array, owned by the caller. Stage diagnostics are
    optional, so normal streaming does not retain every intermediate response.
    """
    def __init__(self,width,height,frame_interval_s=0.05,backend='numpy',device='cpu',config=None):
        if type(width) is not int or type(height) is not int or min(width,height) < 19:
            raise ValueError('V1 source-compatible profile requires integer dimensions >=19')
        if not isfinite(frame_interval_s) or frame_interval_s <= 0:
            raise ValueError('positive finite frame interval required')
        self.width,self.height,self.frame_interval_s = width,height,frame_interval_s
        self.config = config or V1Config()
        self.ops = ArrayOps(backend,device)
        self.edge = self.ops.array(edge_maps(width,height))
        # Match the source CPU double computation, including float32 kernel literals.
        norm = np.asarray(NORM,np.float32).astype(np.float64)
        edge = edge_maps(width,height,dtype=np.float64)
        support = self.ops_support(edge*edge,norm)
        self.compensation = self.ops.array((norm.sum()**2)/support,double=True)
        dirs = np.asarray(ORIENTATIONS,np.float32)
        self.terms = []
        for ot in range(4):
            for oy in range(4-ot):
                ox = 3-oy-ot
                powers = []
                for row,order in zip(dirs,(ox,oy,ot)):
                    p = np.ones(28,np.float32)
                    for _ in range(order):
                        p = p*row
                    powers.append(p)
                coeff = (powers[0]*powers[1])*powers[2]
                multiple = 6//factorial(ox)//factorial(oy)//factorial(ot)
                self.terms.append((ox,oy,ot,multiple,self.ops.array(coeff[:,None,None])))
        self.reset()

    @staticmethod
    def ops_support(edge_squared,kernel):
        ops = ArrayOps()
        return ops.correlate(ops.correlate(edge_squared,kernel,-1),kernel,-2)

    def reset(self):
        self.history = self.ops.zeros((9,self.height,self.width))
        self.frame_index = 0
        self.last_timestamp_s = None

    def provenance(self):
        return {'source':'python_motion_energy','source_sha256':SOURCE_SHA256,
                'config':asdict(self.config),'backend':self.ops.backend,'device':self.ops.device,
                'layout':'SFYX','shape':[3,28,self.height,self.width],
                'frame_interval_s':self.frame_interval_s,'history_frames':9,
                'scale_smoothing':False,'complex_spatial_pooling':False,
                'norm_edge_compensation':True,'rgb_weights':[0.1299,0.5870,0.1140],
                'spike_sampling':False,'time_reference':'frame_start; available after processing frame',
                'filter_delay':'not removed; compare identical frame indices',
                'limitations':['frame/RGB input only; no event/OMS entry point',
                               'source compatibility still requires actual CUDA-library comparison']}

    def step(self,frame,timestamp_s,return_stages=False):
        if not isfinite(timestamp_s) or timestamp_s < 0:
            raise ValueError('finite nonnegative timestamp required')
        if self.last_timestamp_s is not None:
            interval = timestamp_s-self.last_timestamp_s
            if abs(interval-self.frame_interval_s) > max(1e-9,self.frame_interval_s*1e-6):
                raise ValueError('V1 requires fixed cadence; reset or explicitly resample input')
        frame = np.asarray(frame)
        if frame.dtype != np.uint8 or frame.shape not in ((self.height,self.width),(self.height,self.width,3)):
            raise ValueError('frame must be uint8 HxW grayscale or HxWx3 interleaved RGB')
        if self.ops.torch:
            with self.ops.torch.inference_mode():
                result = self._step(frame,return_stages)
        else:
            result = self._step(frame,return_stages)
        self.last_timestamp_s = timestamp_s
        self.frame_index += 1
        return result

    def _step(self,frame,return_stages):
        op,cfg = self.ops,self.config
        values = op.array(frame)
        if frame.ndim == 3:
            gray = ((values[...,0]*0.1299+values[...,1]*0.5870)+values[...,2]*0.1140)*(1.0/255.0)
        else:
            gray = op.float32(op.float64(values)*0.00392156863)
        self.history[:-1] = op.copy(self.history[1:])
        self.history[-1] = gray
        smoothed = self.history
        for axis in (-1,-2,0):
            smoothed = op.correlate(smoothed,GAUSS,axis)
        base = op.zeros((28,self.height,self.width))
        previous_ot,temporal = None,None
        for ox,oy,ot,multiple,coeff in self.terms:
            if ot != previous_ot:
                temporal = op.correlate(smoothed,DIFF[ot],0) if ot else smoothed
                previous_ot = ot
            term = temporal
            if ox:
                term = op.correlate(term,DIFF[ox],-1)
            if oy:
                term = op.correlate(term,DIFF[oy],-2)
            base += (term[4]*multiple)[None,:,:]*coeff
        # source condition is scale>4: all three scale blocks share base.
        linear = base[None,:,:,:]*self.edge[:,None,:,:]
        scaled = op.float32(op.float64(linear)*cfg.linear_gain)
        rectified = op.float32(op.float64(scaled*scaled)*cfg.rect_gain)
        # CUDA mean3 adds channels sequentially.
        population = op.zeros((3,self.height,self.width))
        for f in range(28):
            population += rectified[:,f]
        population = population/28
        population = op.correlate(op.correlate(population,NORM,-1),NORM,-2)
        population = population*cfg.norm_strength
        population = op.float32(op.float64(population)*self.compensation)
        before = rectified*cfg.complex_gain
        rates = op.float32(op.float64(before)/(op.float64(population[:,None])+cfg.c50**2))
        if return_stages:
            return {'linear':linear,'rectified':rectified,'normalization_population':population,
                    'before_normalization':before,'rates':rates}
        return rates
