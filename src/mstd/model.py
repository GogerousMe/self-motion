"""One stateful image -> V1 -> MT -> MSTd model; recording is separate."""
from contextlib import nullcontext
from copy import deepcopy
import json
from math import isfinite
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .data import V1Spec
from .mstd import horizontal_centers
from .projection import load_projection
from .tensor_pipeline import DirectionMT,TensorMSTd
from .v1 import ArrayOps,MotionEnergyV1
from .neural import SpikeConfig,DelayLine,finite
from .spiking_mt import PythonSpikingMT

DEFAULT_CONFIG = {
    'schema':'mstd-model-v1',
    'input':{'kind':'stimulus_dat','path':''},
    'output':'experiments/scratch/manual__model__r01',
    'backend':'numpy','device':'cpu','frame_ms':50,'max_frames':None,
    'projection_config':'configs/v1_projection.json','projection':'motionProj1',
    'intrinsics':None,'mt_radius':1,'mstd_template_count':9,
    'mt_backend':'rate',
    'spiking':SpikeConfig().description(),
    'carlsim':{'executable':'','mode':'cpu','seed':42,'v1_rate_scale':1.0,'inhibition_enabled':True,'callback_layout':'legacy_mapped'},
    'layers':{'v1':True,'mt':True,'mstd':True},
    'monitors':{'v1':True,'mt_projection':True,'mt_pooling':True,
                'mt_normalization':True,'mstd':True,'summary':True,'every_n_frames':1,
                'spikes':False,'neural_state':False},
    'progress_every':20,
}


def model_config(value=None):
    """Merge explicit settings into defaults and reject dependency mistakes."""
    config=deepcopy(DEFAULT_CONFIG)
    value=value or {}
    unknown=set(value)-set(config)
    if unknown:raise ValueError(f'unknown model settings: {sorted(unknown)}')
    for key,item in value.items():
        if key in ('input','layers','monitors','carlsim','spiking'):
            if not isinstance(item,dict) or set(item)-set(config[key]):
                raise ValueError(f'invalid {key} settings')
            config[key].update(item)
        else:config[key]=item
    if config['schema']!='mstd-model-v1':raise ValueError('unsupported model configuration')
    if config['mt_backend'] not in ('rate','carlsim','python_spiking'):raise ValueError('unsupported MT backend')
    if config['mt_backend']=='python_spiking':
        neural=SpikeConfig(**config['spiking'])
        ticks=config['frame_ms']/neural.dt_ms
        if not isfinite(ticks) or abs(ticks-round(ticks))>1e-8 or ticks<1:
            raise ValueError('frame_ms must be an integer multiple of spiking.dt_ms')
        if config['mt_radius']!=0 or not config['layers']['mt']:
            raise ValueError('Python spiking MT requires mt_radius=0 and MT enabled')
    elif config['monitors']['spikes'] or config['monitors']['neural_state']:
        raise ValueError('neural monitors require python_spiking backend')
    if config['mt_backend']=='carlsim':
        native=config['carlsim']
        if not config['layers']['mt'] or config['mt_radius']!=0:
            raise ValueError('CARLsim baseline requires MT enabled and mt_radius=0')
        if native['mode'] not in ('cpu','gpu') or type(native['seed']) is not int or not 0<=native['seed']<2**31:
            raise ValueError('invalid CARLsim mode/seed')
        scale=native['v1_rate_scale']
        if isinstance(scale,bool) or not isinstance(scale,(int,float)) or not isfinite(scale) or scale<=0:
            raise ValueError('positive finite v1_rate_scale required')
        if not isinstance(native['executable'],str):raise ValueError('executable must be a path string')
        if type(native['inhibition_enabled']) is not bool:raise ValueError('inhibition_enabled must be boolean')
        if native['callback_layout'] not in ('legacy_mapped','standard'):raise ValueError('invalid callback_layout')
        if type(config['frame_ms']) is not int or config['frame_ms']<=0:
            raise ValueError('CARLsim requires positive integer frame_ms')
    for key in ('layers','monitors'):
        for name,item in config[key].items():
            if name!='every_n_frames' and type(item) is not bool:
                raise ValueError(f'{key}.{name} must be true or false')
    stride=config['monitors']['every_n_frames']
    if type(stride) is not int or stride<=0:raise ValueError('positive integer monitor stride required')
    if not any(config['layers'].values()):raise ValueError('enable at least one processing layer')
    if config['layers']['mstd'] and not config['layers']['mt']:
        raise ValueError('MSTd requires MT; disable MSTd when MT is disabled')
    kind=config['input']['kind']
    if kind not in ('stimulus_dat','v1_recording'):raise ValueError('unsupported input kind')
    if kind=='stimulus_dat' and not config['layers']['v1']:
        raise ValueError('image input requires V1; to disable V1 supply input.kind=v1_recording')
    if kind=='v1_recording' and config['layers']['v1']:
        raise ValueError('existing V1 rates require layers.v1=false; do not filter rates as images')
    dt=config['frame_ms']
    if isinstance(dt,bool) or not isinstance(dt,(int,float)) or not isfinite(dt) or dt<=0:
        raise ValueError('positive finite frame_ms required')
    for key in ('mt_radius','progress_every'):
        if type(config[key]) is not int or config[key]<0:raise ValueError(f'invalid {key}')
    if type(config['mstd_template_count']) is not int or config['mstd_template_count']<2:
        raise ValueError('at least two MSTd templates required')
    if config['max_frames'] is not None and (type(config['max_frames']) is not int or config['max_frames']<=0):
        raise ValueError('positive max_frames required')
    return config


def load_model_config(path):
    return model_config(json.loads(Path(path).read_text(encoding='utf-8')))


class VisionModel:
    """All enabled layers share one environment and one selected array device.

    Layer switches control computation. Monitor switches belong to the runner
    and do not change this model's state or outputs.
    """
    def __init__(self,width,height,config=None):
        self.config=model_config(config)
        if self.config['mt_backend']=='carlsim':
            raise ValueError('CARLsim is an offline backend; use simulate(), not VisionModel.step()')
        self.width,self.height=width,height
        self.interval=self.config['frame_ms']/1000
        layers=self.config['layers']
        self.v1=MotionEnergyV1(width,height,self.interval,self.config['backend'],self.config['device']) if layers['v1'] else None
        self.ops=self.v1.ops if self.v1 else ArrayOps(self.config['backend'],self.config['device'])
        self.mt,self.mstd,self.projection=None,None,None
        self.intrinsics=self.config['intrinsics']
        self.centers=()
        if layers['mt']:
            self.projection=load_projection(self.config['projection_config'],self.config['projection'])
            self.mt=(PythonSpikingMT(self.ops,self.projection,width,height,SpikeConfig(**self.config['spiking']))
                     if self.config['mt_backend']=='python_spiking' else
                     DirectionMT(self.ops,self.projection,width,height,self.config['mt_radius']))
        if isinstance(self.intrinsics,str):
            self.intrinsics=json.loads(Path(self.intrinsics).read_text(encoding='utf-8'))
        if layers['mstd']:
            count=self.config['mstd_template_count']
            if self.intrinsics is not None:
                spec=V1Spec(width,height,self.projection.directions_rad,**self.intrinsics)
                self.centers=tuple(horizontal_centers(spec,np.linspace(-20,20,count)))
            else:
                # Pixel-space template activities remain meaningful without camera calibration.
                # Never invent focal lengths or publish uncalibrated heading angles.
                spec=SimpleNamespace(width=width,height=height,fx_px=None,fy_px=None,
                                     cx_px=(width-1)/2,cy_px=(height-1)/2)
                self.centers=tuple((float(x),(height-1)/2) for x in np.linspace(0,width-1,count))
            self.mstd=TensorMSTd(self.ops,spec,self.centers)
        self.feedback_controller=None
        self.spike_observer=None
        self.feedback_delay=(DelayLine(self.ops,(4,8,height,width),self.config['spiking']['feedback_delay_ticks'])
                             if self.config['mt_backend']=='python_spiking' else None)
        self.reset()

    def reset(self):
        if self.v1:self.v1.reset()
        if self.mstd:self.mstd.reset()
        if self.mt and hasattr(self.mt,'reset'):self.mt.reset()
        if self.feedback_delay:self.feedback_delay.reset()
        self.ticks=0
        self.last_timestamp=None
        self.frames=0

    def set_feedback(self,controller=None):
        """controller(context) -> optional dict of device arrays shaped 8,H,W.

        exc_current/inh_current modulate E/I neurons in Izhikevich current units.
        feedback_exc_events/feedback_inh_events deliver weighted nonnegative
        synaptic events onto MT-E with their own exponential conductance states.

        Called at tick END with owned snapshots. Currents arrive after the
        configured delay, so the controller cannot affect the tick it observes.
        This interface does not assert any specific biological feedback rule.
        """
        if self.feedback_delay is None:raise ValueError('feedback requires python_spiking')
        if controller is not None and not callable(controller):raise ValueError('controller must be callable')
        self.feedback_controller=controller

    def set_spike_observer(self,observer=None):
        if self.feedback_delay is None:raise ValueError('spike observer requires python_spiking')
        if observer is not None and not callable(observer):raise ValueError('observer must be callable')
        self.spike_observer=observer

    def _spiking_frame(self,raw,timestamp_s):
        op=self.ops
        self.mt.set_rates(raw)
        dt_ms=self.config['spiking']['dt_ms'];dt=dt_ms/1000
        count=round(self.config['frame_ms']/dt_ms)
        totals={k:op.zeros(shape) for k,shape in (('v1',self.mt.input_shape),('exc',self.mt.shape),('inh',self.mt.shape))}
        mst=None
        for i in range(count):
            arriving=self.feedback_delay.read()
            spikes=self.mt.tick(arriving)
            mt=self.mt.readout()
            # Neural receivers can consume events directly; the temporary rate
            # MSTd adapter ignores these extra fields. Receivers own their delays.
            mt.update(spikes=spikes['exc'],inh_spikes=spikes['inh'])
            mst=self.mstd.step(mt,dt) if self.mstd else None
            end=timestamp_s+(i+1)*dt
            outgoing=op.zeros((4,*self.mt.shape))
            if self.feedback_controller:
                context={'tick':self.ticks,'timestamp_s':end,
                         'mt':{k:op.copy(v) if hasattr(v,'shape') else v for k,v in mt.items()},
                         'spikes':{k:op.copy(v) for k,v in spikes.items()},'mstd':deepcopy(mst)}
                response=self.feedback_controller(context)
                if response is not None:
                    keys=('exc_current','inh_current','feedback_exc_events','feedback_inh_events')
                    if not isinstance(response,dict) or set(response)-set(keys):
                        raise ValueError('feedback must return known current/synaptic event arrays or None')
                    for j,key in enumerate(keys):
                        if key in response:
                            value=op.array(response[key])
                            if (tuple(value.shape)!=self.mt.shape or not finite(op,value)
                                or (j>=2 and not bool((value>=0).all()))):raise ValueError('invalid feedback current/event')
                            outgoing[j]=value
            self.feedback_delay.write(outgoing)
            for k in totals:totals[k]+=spikes[k]
            if self.spike_observer:self.spike_observer(self.ticks,end,{k:op.copy(v) for k,v in spikes.items()})
            self.ticks+=1
        result={**mt,'direction_rates':totals['exc']/self.interval,'inh_rates':totals['inh']/self.interval,
                'exc_counts':totals['exc'],'inh_counts':totals['inh'],'v1_counts':totals['v1']}
        if self.config['monitors']['neural_state']:result.update(self.mt.diagnostics())
        return result,mst

    def step(self,value,timestamp_s):
        if not isfinite(timestamp_s) or timestamp_s<0:raise ValueError('invalid timestamp')
        if self.last_timestamp is not None and abs(timestamp_s-self.last_timestamp-self.interval)>max(1e-9,self.interval*1e-6):
            raise ValueError('fixed frame cadence required')
        with self.ops.torch.inference_mode() if self.ops.torch else nullcontext():
            raw=self.v1.step(value,timestamp_s) if self.v1 else self.ops.array(value)
            if tuple(raw.shape)!=(3,28,self.height,self.width):raise ValueError('expected SFYX V1 rates')
            if self.config['mt_backend']=='python_spiking':mt,mst=self._spiking_frame(raw,timestamp_s)
            else:
                mt=self.mt.step(raw) if self.mt else None
                mst=self.mstd.step(mt,self.interval) if self.mstd else None
        self.frames+=1
        self.last_timestamp=timestamp_s
        return {'timestamp_s':timestamp_s,'available_timestamp_s':timestamp_s+self.interval,
                'v1':raw,'mt':mt,'mstd':mst}

    def description(self):
        return {'layers':self.config['layers'],'backend':self.ops.backend,'device':self.ops.device,
                'mt_backend':self.config['mt_backend'],
                'neural':({**self.config['spiking'],'ticks':self.ticks,
                           'mstd_type':'continuous-rate adapter advanced every neural tick',
                           'feedback':'delayed E/I current modulation or synaptic events onto MT-E; disabled by default',
                           'spike_timestamps':'tick end; V1 multiple Poisson events retained',
                           'synapses':'fast exponential E/I conductance, project units; no slow receptors'}
                          if self.config['mt_backend']=='python_spiking' else None),
                'width':self.width,'height':self.height,'frame_interval_s':self.interval,
                'v1':self.v1.provenance() if self.v1 else {'source':'replayed_continuous_v1'},
                'mstd_template_centers_px':[list(c) for c in self.centers],
                'mstd_template_coordinates':'camera-bearing grid' if self.intrinsics is not None else 'pixel grid',
                'intrinsics_px':self.intrinsics,'bearing_calibrated':self.intrinsics is not None}
