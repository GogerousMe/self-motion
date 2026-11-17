"""MSTd dynamics ablations on ideal fields or timestamped saved MT samples."""
import argparse
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from mstd.calibration import file_sha256
from mstd.data import V1Spec
from mstd.mstd import horizontal_centers
from mstd.readout import ReadoutConfig, heading_readout
from mstd.temporal import temporal_variants, ideal_sequence, response_metrics
from mstd.tensor_pipeline import TensorMSTd
from mstd.v1 import ArrayOps
from mstd.validity import ValidityConfig
from mstd.v1_recording import dump_json
from run_mt_spatial_controls import append_index, register_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['numpy','torch'], default='torch')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--recording', type=Path)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--frames', type=int, default=100)
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    configs = temporal_variants()
    validity, readout = ValidityConfig(mode='geometry_support'), ReadoutConfig(mode='local_quadratic')
    dt = .05
    source = None
    if args.recording:
        recording = args.recording.resolve()
        report = json.loads((recording/'model_report.json').read_text(encoding='utf-8'))
        meta_path = recording/'monitors/mt_readout_activity/metadata.json'
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        count, directions, height, width = meta['shape']
        if (not report['complete'] or not meta['complete'] or directions != 8 or count != report['frames']
                or meta['units'] != 'dimensionless' or meta['layout'] != 'TDYX'
                or meta['dtype'] != 'float32-le' or meta['sample_indices'] != list(range(count))):
            raise ValueError('complete every-frame dimensionless MT readout required')
        dt = report['model']['frame_interval_s']
        times = np.asarray(meta['timestamps_s'])
        if not np.allclose(times, (np.arange(count)+1)*dt, atol=1e-8, rtol=0):
            raise ValueError('replay requires frame-end MT timestamps')
        source = meta_path.parent/meta['file']
        if source.stat().st_size != count*8*height*width*4:
            raise ValueError('MT binary size mismatch')
        camera = report['model']['intrinsics_px']
        spec = (V1Spec(width,height,tuple(meta['directions_rad']),**camera) if camera else
                SimpleNamespace(width=width,height=height,fx_px=None,fy_px=None,cx_px=(width-1)/2,cy_px=(height-1)/2))
        cases = ['recorded_mt']
    else:
        if args.width < 8 or args.frames < 80:
            parser.error('ideal protocol requires width>=8 and frames>=80')
        width = height = args.width
        count = args.frames
        spec = V1Spec(width,height,tuple(np.arange(8)*np.pi/4),width,width,(width-1)/2,(height-1)/2)
        times = np.arange(count)*dt
        cases = ['hold','step','sweep','occlusion','uniform','noise_stop']
    centers = (horizontal_centers(spec,np.linspace(-20,20,9)) if spec.fx_px is not None else
               [(float(x),spec.cy_px) for x in np.linspace(0,width-1,9)])
    ticks = round(dt/.001)
    if ticks < 1 or abs(ticks*.001-dt)>1e-9:
        raise ValueError('sample interval must be an integer multiple of1ms')
    plan = dict(mode='saved_mt_replay' if source else 'ideal_mt', variants={k:asdict(v) for k,v in configs.items()},
                cases=cases, frames=count, width=width,height=height,frame_s=dt,tick_s=.001,
                validity=asdict(validity),readout=asdict(readout),seed=args.seed,
                intrinsics_px={k:getattr(spec,k,None) for k in ('fx_px','fy_px','cx_px','cy_px')},
                command=[sys.executable,*sys.argv],
                input_semantics='dimensionless direction-evidence vectors',
                replay_semantics='hold each saved frame-end sample from its availability time for one interval')
    if args.plan_only:
        print(json.dumps(plan,indent=2));return
    op = ArrayOps(args.backend,args.device)
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    mode = 'replay' if source else 'ideal'
    device = 'cuda' if op.device.startswith('cuda') else 'cpu'
    rid = f'{stamp}__mstd_temporal_{mode}_{width}x{height}_f{count:06d}__py_rate_{device}__r01'
    out = ROOT/'experiments/validation'/rid
    out.mkdir(parents=True,exist_ok=False)
    dump_json(out/'plan.json',plan)
    sources = ['src/mstd/temporal.py','src/mstd/mstd.py','src/mstd/tensor_pipeline.py','src/mstd/readout.py',
               'src/mstd/validity.py','src/mstd/model.py','tools/run_mstd_temporal_controls.py']
    artifact = dict(schema='mstd-artifact-v1',id=rid,kind='validation',status='running',
                    parent_run_ids=[],stimulus_ids=[],input_paths=[],source_hashes={p:file_sha256(ROOT/p) for p in sources},
                    notes='MSTd continuous-rate temporal ablations with fixed inputs and geometry acceptance.')
    saved = None
    summaries = []
    try:
        if source:
            saved = np.memmap(source,mode='r',dtype='<f4',shape=(count,8,height,width))
            if not np.isfinite(saved).all() or (saved < 0).any():
                raise ValueError('MT activity must be finite and nonnegative')
            artifact.update(input_paths=[str(source)],input_sha256=file_sha256(source))
            parent = args.recording/'artifact.json'
            if parent.is_file():artifact['parent_run_ids']=[json.loads(parent.read_text())['id']]
        for case in cases:
            folder = out/case
            folder.mkdir()
            models = {name:TensorMSTd(op,spec,centers,config=cfg,validity_config=validity,readout_config=readout)
                      for name,cfg in configs.items()}
            rows = {name:[] for name in models}
            streams = {name:(folder/(name+'.jsonl')).open('x',encoding='utf-8') for name in models}
            fixture = None
            if not source:
                fixture = (folder/'input_vectors.f32').open('xb')
                iterator = ideal_sequence(spec,case,count,args.seed)
            else:
                units = np.stack((np.cos(spec.directions_rad),np.sin(spec.directions_rad)),axis=1) if spec.fx_px is not None else np.stack((np.cos(meta['directions_rad']),np.sin(meta['directions_rad'])),axis=1)
                iterator = ((np.einsum('dyx,dc->cyx',saved[i],units).astype(np.float32),
                             dict(frame=i,heading_deg=None,phase='saved_sample')) for i in range(count))
            try:
                for vectors, truth in iterator:
                    if not np.isfinite(vectors).all():raise ValueError('nonfinite MT vector')
                    if fixture:fixture.write(vectors.astype('<f4').tobytes())
                    signal = dict(vectors=op.array(vectors.copy()),evidence=float(np.linalg.norm(vectors,axis=0).mean()))
                    for name, model in models.items():
                        for tick in range(ticks):
                            state = model.step(signal,.001,return_state=tick==ticks-1)
                        candidate = heading_readout(state['activity'],centers,spec,state['winner_index'],True,readout)['alpha_x_deg']
                        row = dict(**state,**truth,input_timestamp_s=float(times[truth['frame']]),
                                   estimate_timestamp_s=float(times[truth['frame']]+dt),candidate_heading_deg=candidate)
                        streams[name].write(json.dumps(row,allow_nan=False)+'\n');rows[name].append(row)
            finally:
                for stream in streams.values():stream.close()
                if fixture:fixture.close()
            if not source:
                path = folder/'input_vectors.f32'
                dump_json(folder/'input.json',dict(shape=[count,2,height,width],layout='TCYX',dtype='float32-le',
                    units='dimensionless',frame_s=dt,fx_px=spec.fx_px,fy_px=spec.fy_px,cx_px=spec.cx_px,cy_px=spec.cy_px,
                    truth=[dict(frame=r['frame'],heading_deg=r['heading_deg'],phase=r['phase']) for r in rows['reference']],
                    sha256=file_sha256(path)))
                artifact['input_paths'].append(path.relative_to(ROOT).as_posix())
                append_index(ROOT/'input/registry.csv',dict(path=path.relative_to(ROOT).as_posix(),width=width,height=height,
                    frames=count,channels=2,sha256=file_sha256(path),scope='fixture'),
                    ['path','width','height','frames','channels','sha256','scope'])
            for name in models:
                metrics = response_metrics(rows[name],dt,case=case) if not source else dict(frames=count,
                    bearing_valid_frames=sum(r['bearing_valid'] for r in rows[name]),
                    geometry_valid_frames=sum(r['valid'] for r in rows[name]),replay_additional_hold_s=dt)
                summaries.append(dict(case=case,variant=name,**metrics))
            dump_json(out/'summary.json',summaries)
            print('COMPLETE',case,flush=True)
        artifact['status']='completed'
    except BaseException:
        artifact['status']='failed';raise
    finally:
        if saved is not None:saved._mmap.close()
        register_run(out,artifact)
    print('OUTPUT',out.relative_to(ROOT).as_posix(),flush=True)


if __name__=='__main__':
    main()
