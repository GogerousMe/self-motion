"""Real CARLsim offline MT backend; rate readout feeds the existing MSTd.

No approximate native-engine fallback. Explicit callback ABI handles the user's
patched legacy kernel as well as the standard CARLsim interface.
"""
from contextlib import ExitStack
from copy import deepcopy
from itertools import islice
import json
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np

from .calibration import file_sha256
from .carlsim import iter_bins, read_header
from .model import VisionModel
from .monitors import ArrayMonitor
from .projection import load_projection
from .v1_recording import dump_json, raw_info, iter_raw, RawWriter


def spike_readout(rates, directions_rad):
    """Causal bin-end adapter, NOT an MT neural normalization circuit."""
    rates=np.asarray(rates,dtype=np.float32)
    if rates.ndim!=3 or rates.shape[0]!=8 or not np.isfinite(rates).all() or (rates<0).any():
        raise ValueError('expected nonnegative finite 8,Y,X observed rates')
    if len(directions_rad)!=8 or not np.isfinite(directions_rad).all():
        raise ValueError('eight finite directions required')
    activity=rates/(1.0+rates.sum(axis=0,dtype=np.float64))
    units=np.array([(np.cos(a),np.sin(a)) for a in directions_rad])
    vectors=np.einsum('dyx,dc->cyx',activity,units)
    evidence=float(np.hypot(vectors[0],vectors[1]).mean())
    return dict(activity=activity.astype(np.float32),vectors=vectors.astype(np.float32),
                evidence=evidence,direction_rates=rates,pooled_rates=rates)


def spike_frames(root,width,height,frames,frame_ms):
    """Exhaust every original DAT, including V1, to validate event ranges/tails."""
    duration=frames*frame_ms
    with ExitStack() as stack:
        streams={}
        for group,count in (('v1',84),('exc',8),('inh',8)):
            path=Path(root)/f'{group}.dat'
            header=read_header(path)
            if (header.width,header.height,header.depth)!=(width,height,count):
                raise ValueError('CARLsim group grid mismatch')
            iterator=iter_bins(path,duration,frame_ms)
            stack.callback(iterator.close)
            streams[group]=(iterator,count)
        for index in range(frames):
            result={}
            for group,(it,depth) in streams.items():
                start,span,values=next(it)
                if (start,span)!=(index*frame_ms,frame_ms):raise ValueError('misaligned spike windows')
                result[group]=np.asarray(values,dtype=np.float32).reshape(depth,height,width)*(1000.0/frame_ms)
            yield index,result
        # Resume after final yield: iter_bins then checks any out-of-duration event.
        for it,_ in streams.values():
            if next(it,None) is not None:raise ValueError('unexpected extra spike window')


def simulate_carlsim(config):
    from .simulation import simulate
    native=config['carlsim']
    executable=Path(native['executable']).resolve()
    if not native['executable'] or not executable.is_file():
        raise FileNotFoundError('build native/carlsim_mt and set carlsim.executable; no rate fallback')
    projection=load_projection(config['projection_config'],config['projection'])
    if projection.scale_weights!=(1.,1.,1.) or projection.gain!=1.:
        raise ValueError('CARLsim baseline requires unit projection scale weights and gain')
    output=Path(config['output']).resolve()
    # Let existing input validation run before it creates the new parent output.
    frontend=deepcopy(config)
    frontend.update(mt_backend='rate',output=str(output/'frontend'),layers=dict(v1=config['layers']['v1'],mt=False,mstd=False))
    frontend['monitors']={k:False for k in ('v1','mt_projection','mt_pooling','mt_normalization','mstd','summary')}
    frontend['monitors'].update(v1=True,every_n_frames=1)
    if output.exists():raise FileExistsError(output)
    started=perf_counter()
    raw=output/'frontend/monitors/v1'
    if config['input']['kind']=='v1_recording':
        source=Path(config['input']['path'])
        meta=raw_info(source)
        if meta.get('monitor_every_n_frames',1)!=1 or abs(meta['frame_interval_s']-config['frame_ms']/1000)>1e-9:
            raise ValueError('V1 replay requires complete every-frame recording at frame_ms cadence')
        provenance={**meta.get('provenance',{}),'input_kind':'v1_recording',
                    'input_path':str(source),'input_sha256':file_sha256(source/'rates.f32')}
        writer=RawWriter(raw,meta['width'],meta['height'],meta['frame_interval_s'],provenance)
        success=False
        try:
            with ExitStack() as stack:
                iterator=iter_raw(source);stack.callback(iterator.close)
                for value in islice(iterator,config['max_frames']):writer.append(value)
            success=True
        finally:writer.close(success)
        front_report={'input':provenance}
    else:
        front_report=simulate(frontend)
    info=raw_info(raw)
    frames,width,height=info['frames'],info['width'],info['height']
    if frames*config['frame_ms']>=2**31:raise ValueError('CARLsim duration exceeds int32 time')
    weights=output/'weights.f32'
    np.asarray(projection.weights,dtype='<f4').tofile(weights)
    native_dir=output/'spikes'
    native_dir.mkdir()
    dump_json(output/'model_config.json',config)
    command=[str(executable),str(raw/'rates.f32'),str(weights),str(width),str(height),str(frames),
             str(config['frame_ms']),str(native['seed']),native['mode'],str(native['v1_rate_scale']),
             str(int(native['inhibition_enabled'])),native['callback_layout']]
    # Keep diagnostics out of the binary streams and preserve failed-run logs.
    with (output/'carlsim.log').open('x',encoding='utf-8') as log:
        subprocess.run(command,cwd=native_dir,stdout=log,stderr=subprocess.STDOUT,check=True)
    capture=json.loads((native_dir/'native_report.json').read_text(encoding='utf-8'))
    expected={'schema':'mstd-carlsim-native-v1','complete':True,'width':width,'height':height,
              'frames':frames,'frame_ms':config['frame_ms'],'seed':native['seed'],'mode':native['mode']}
    if any(capture.get(k)!=v for k,v in expected.items()):raise ValueError('native run metadata mismatch')
    edges=width*height*(3*np.count_nonzero(np.asarray(projection.weights))+8*int(native['inhibition_enabled']))
    if capture.get('connections')!=edges:raise ValueError('native connection count mismatch')
    # Instantiate only the existing geometry/MSTd; never recompute continuous MT.
    read_config=deepcopy(config)
    read_config.update(mt_backend='rate',input=dict(kind='v1_recording',path=str(raw)),layers=dict(v1=False,mt=True,mstd=config['layers']['mstd']))
    model=VisionModel(width,height,read_config)
    dt=config['frame_ms']/1000
    dirs={'directions_rad':list(projection.directions_rad),'coordinates':'x-right_y-down',
          'timestamp_reference':'bin_end','readout':'observed counts / full frame duration'}
    monitors={}
    last=None
    read_start=perf_counter()
    with ExitStack() as stack:
        for name,units in (('mt_exc_rates','Hz'),('mt_inh_rates','Hz'),('mt_readout_activity','dimensionless')):
            monitor=ArrayMonitor(output/'monitors'/name,(8,height,width),'DYX',units,dirs)
            monitors[name]=monitor
        estimates=stack.enter_context((output/'estimates.jsonl').open('x',encoding='utf-8'))
        iterator=spike_frames(native_dir,width,height,frames,config['frame_ms'])
        stack.callback(iterator.close)
        success=False
        totals=dict(v1=0.,exc=0.,inh=0.)
        try:
            for index,values in iterator:
                mt=spike_readout(values['exc'],projection.directions_rad)
                for k in totals:totals[k]+=float(values[k].sum(dtype=np.float64))*dt
                timestamp=(index+1)*dt
                if index%config['monitors']['every_n_frames']==0:
                    for name,value in (('mt_exc_rates',values['exc']),('mt_inh_rates',values['inh']),('mt_readout_activity',mt['activity'])):
                        monitors[name].append(value,index,timestamp)
                if model.mstd:
                    last=model.mstd.step({**mt,'vectors':model.ops.array(mt['vectors'])},dt)
                    estimates.write(json.dumps(dict(frame=index,estimate_timestamp_s=timestamp,**last),allow_nan=False)+'\n')
            success=True
        finally:
            for monitor in monitors.values():monitor.close(success)
    report=dict(schema='mstd-spiking-simulation-v1',complete=True,frames=frames,
                model={**model.description(),'layers':config['layers'],'mt_backend':'carlsim','mt_neural_normalization':False,
                       'inhibition_enabled':native['inhibition_enabled'],
                       'readout':'bin-end observed rates; adapter r/(1Hz+sum(r))'},
                input=front_report['input'],native=capture,last_mstd=last,event_counts={k:int(round(v)) for k,v in totals.items()},
                readout_seconds=perf_counter()-read_start,wall_seconds=perf_counter()-started,
                hashes=dict(executable=file_sha256(executable),weights=file_sha256(weights),
                            projection_config=file_sha256(config['projection_config']),v1_rates=file_sha256(raw/'rates.f32')),
                build_info=json.loads((executable.parent/'build_info.json').read_text(encoding='utf-8'))
                    if (executable.parent/'build_info.json').is_file() else None,
                command=command,limitations=[
                    'real CARLsim dynamics; historical snapshot/library equivalence not yet established',
                    'callback ABI must match the linked library; historical comparison requires identical inputs and engine build',
                    'continuous MSTd with rate adapter, not an all-spiking network',
                    'offline two-pass runner; no online FPS claim',
                    'CARLsim library defaults/build must be recorded separately for reproducible neuron dynamics'])
    dump_json(output/'model_report.json',report)
    return report
