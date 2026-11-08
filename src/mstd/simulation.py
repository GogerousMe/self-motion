"""Unified offline project runner; future live sources can call VisionModel.step."""
import json
from itertools import islice
from pathlib import Path
from time import perf_counter

from .calibration import file_sha256,inspect_stimulus
from .model import VisionModel,model_config
from .monitors import ModelMonitors
from .v1_recording import dump_json,iter_raw,iter_stimulus,raw_info


def simulate(settings):
    config=model_config(settings)
    if config['mt_backend']=='carlsim':
        from .spiking import simulate_carlsim
        return simulate_carlsim(config)
    path=Path(config['input']['path'])
    if not config['input']['path']:raise ValueError('provide input.path or --stimulus')
    replay=config['input']['kind']=='v1_recording'
    if replay:
        info=raw_info(path)
        if info.get('monitor_every_n_frames',1)!=1:
            raise ValueError('V1 replay requires every-frame recording; sparse monitors cannot reproduce full history')
        if abs(info['frame_interval_s']-config['frame_ms']/1000)>1e-9:
            raise ValueError('V1 recording interval disagrees with frame_ms')
        source_hash=file_sha256(path/'rates.f32')
    else:
        info=inspect_stimulus(path)
        source_hash=file_sha256(path)
    count=min(info['frames'],config['max_frames']) if config['max_frames'] else info['frames']
    init_started=perf_counter()
    model=VisionModel(info['width'],info['height'],config)
    initialization_seconds=perf_counter()-init_started
    output=Path(config['output'])
    output.mkdir(parents=True,exist_ok=False)
    dump_json(output/'model_config.json',config)
    provenance=model.v1.provenance() if model.v1 else dict(info.get('provenance',{}))
    provenance.update({'input_kind':config['input']['kind'],'input_path':str(path),'input_sha256':source_hash,
                       'complete_stimulus':count==info['frames']})
    if not replay:provenance['stimulus_sha256']=source_hash
    monitors=ModelMonitors(output,model,provenance)
    if config['mt_backend']=='python_spiking' and config['monitors']['spikes']:
        model.set_spike_observer(monitors.record_spikes)
    source=iter_raw(path) if replay else None
    iterator=enumerate(islice(source,count)) if replay else iter_stimulus(path,config['max_frames'])
    success=False
    started=perf_counter()
    compute_seconds=0.0
    last=None
    try:
        for index,value in iterator:
            if index>=count:break
            model.ops.sync()
            begin=perf_counter()
            state=model.step(value,index*model.interval)
            model.ops.sync()
            compute_s=perf_counter()-begin
            compute_seconds+=compute_s
            monitors.record(state,index,compute_s)
            last=state['mstd']
            if config['progress_every'] and (index==0 or (index+1)%config['progress_every']==0 or index+1==count):
                print(f'[model] {index+1}/{count}, elapsed={perf_counter()-started:.2f}s',flush=True)
        success=True
    finally:
        if hasattr(iterator,'close'):iterator.close()
        if source is not None:source.close()
        monitors.close(success)
    elapsed=perf_counter()-started
    report={'schema':'mstd-simulation-v1','complete':True,'frames':model.frames,
            'model':model.description(),'monitors':monitors.description(),'input':provenance,
            'last_mstd':last,'wall_seconds':elapsed,'compute_seconds':compute_seconds,
            'initialization_seconds':initialization_seconds,
            'wall_fps':model.frames/elapsed,'limitations':[
                ('Python MT spikes and fast E/I synapses; MSTd remains a continuous-rate adapter; no CARLsim equivalence claim'
                 if config['mt_backend']=='python_spiking' else 'continuous-rate model; monitors are not membrane-voltage/spike monitors'),
                'without intrinsics, MSTd uses pixel templates and reports no calibrated bearing',
                'CUDA and original-source equivalence still require server validation']}
    dump_json(output/'model_report.json',report)
    return report
