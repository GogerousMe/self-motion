"""Strict, streaming continuous V1 recording and image-DAT readers."""
import json
from math import isfinite
from pathlib import Path

import numpy as np

from .calibration import file_sha256,inspect_stimulus


def dump_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def iter_stimulus(path,max_frames=None):
    """Read uint8 frames for all VisualStimulus categories (types 0 through 6).

    Type names describe the stimulus content, not its pixel dtype. CARLsim's
    readFrameChar reads the same byte layout for grating, movie and compound
    stimuli as it does for pictures. inspect_stimulus validates header/size.
    """
    info = inspect_stimulus(path)
    if max_frames is not None and (type(max_frames) is not int or max_frames<=0):
        raise ValueError('positive max_frames required')
    count = min(info['frames'],max_frames) if max_frames else info['frames']
    shape = (info['height'],info['width'])+((3,) if info['channels']==3 else ())
    size = int(np.prod(shape))
    with Path(path).open('rb') as stream:
        stream.seek(25)
        for i in range(count):
            block = stream.read(size)
            if len(block)!=size:
                raise ValueError('truncated stimulus payload')
            yield i,np.frombuffer(block,dtype=np.uint8).reshape(shape)


class RawWriter:
    def __init__(self,path,width,height,frame_interval_s,provenance):
        self.path = Path(path)
        self.path.mkdir(parents=True,exist_ok=False)
        self.width,self.height,self.interval = width,height,frame_interval_s
        self.provenance,self.frames = provenance,0
        self.stream = (self.path/'rates.f32').open('xb')

    def append(self,rates):
        rates = np.asarray(rates)
        if rates.shape!=(3,28,self.height,self.width) or not np.isfinite(rates).all() or (rates<0).any():
            raise ValueError('invalid SFYX response')
        self.stream.write(rates.astype('<f4',copy=False).tobytes())
        self.frames += 1

    def close(self,success=True):
        self.stream.close()
        if success and self.frames:
            dump_json(self.path/'metadata.json',{
                'schema':'v1-filter-rate-v1','dtype':'float32-le','layout':'TSFYX',
                'encoding':'raw_filter_rates','units':'Hz','width':self.width,'height':self.height,
                'scales':3,'filters':28,'frames':self.frames,'frame_interval_s':self.interval,
                'timestamp_reference':'frame_start','history_initialization':'zeros',
                'provenance':self.provenance})
        # On failure keep the partial binary without metadata: it cannot be read as complete.


def raw_info(path):
    directory = Path(path)
    meta = json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
    required = {'schema':'v1-filter-rate-v1','dtype':'float32-le','layout':'TSFYX',
                'encoding':'raw_filter_rates','units':'Hz','scales':3,'filters':28,
                'timestamp_reference':'frame_start','history_initialization':'zeros'}
    if any(meta.get(k)!=v for k,v in required.items()):
        raise ValueError('unsupported raw V1 recording')
    if any(type(meta.get(k)) is not int or meta[k]<=0 for k in ('width','height','frames')):
        raise ValueError('invalid raw dimensions')
    dt = meta.get('frame_interval_s')
    if not isinstance(dt,(int,float)) or not isfinite(dt) or dt<=0:
        raise ValueError('invalid raw frame interval')
    size = meta['frames']*84*meta['width']*meta['height']*4
    if (directory/'rates.f32').stat().st_size!=size:
        raise ValueError('raw recording binary size mismatch')
    return meta


def iter_raw(path):
    meta = raw_info(path)
    shape = (3,28,meta['height'],meta['width'])
    size = int(np.prod(shape))*4
    with (Path(path)/'rates.f32').open('rb') as stream:
        for _ in range(meta['frames']):
            block = stream.read(size)
            if len(block)!=size:
                raise ValueError('raw recording changed while reading')
            value = np.frombuffer(block,dtype='<f4').reshape(shape)
            if not np.isfinite(value).all() or (value<0).any():
                raise ValueError('nonfinite or negative raw V1 rates')
            yield value
        if stream.read(1):
            raise ValueError('raw recording changed while reading')


def register_cpp_capture(path,stimulus,source_path,library_path):
    """Bind a standalone C++ capture to exact input/source/library hashes."""
    from .v1 import SOURCE_SHA256
    directory = Path(path)
    if (directory/'metadata.json').exists():
        raise FileExistsError('capture already registered')
    capture = json.loads((directory/'capture.json').read_text(encoding='utf-8'))
    info = inspect_stimulus(stimulus)
    if any(type(capture.get(k)) is not int for k in ('width','height','channels','frames')):
        raise ValueError('integer capture dimensions required')
    dt = capture.get('frame_interval_s')
    if not isinstance(dt,(int,float)) or not isfinite(dt) or dt<=0:
        raise ValueError('positive finite capture interval required')
    if any(capture[k]!=info[k] for k in ('width','height','channels')):
        raise ValueError('capture/stimulus dimensions disagree')
    if not 0<capture['frames']<=info['frames']:
        raise ValueError('capture frame count outside stimulus')
    source_hash = file_sha256(source_path)
    if source_hash!=SOURCE_SHA256:
        raise ValueError('source hash differs from port profile; inspect source differences before comparison')
    provenance = {'source':'cpp_motion_energy_continuous','stimulus_sha256':file_sha256(stimulus),
                  'source_sha256':source_hash,'library_sha256':file_sha256(library_path),
                  'source_path':str(source_path),'library_path':str(library_path),
                  'source_profile':'motion-energy-frame-20261028',
                  'complete_stimulus':capture['frames']==info['frames'],
                  'note':'source hash records supplied source; verify library was built from it'}
    meta = {'schema':'v1-filter-rate-v1','dtype':'float32-le','layout':'TSFYX',
            'encoding':'raw_filter_rates','units':'Hz','width':info['width'],'height':info['height'],
            'scales':3,'filters':28,'frames':capture['frames'],'frame_interval_s':capture['frame_interval_s'],
            'timestamp_reference':'frame_start','history_initialization':'zeros','provenance':provenance}
    expected = capture['frames']*84*info['width']*info['height']*4
    if (directory/'rates.f32').stat().st_size!=expected:
        raise ValueError('capture payload size mismatch')
    dump_json(directory/'metadata.json',meta)
    return meta
