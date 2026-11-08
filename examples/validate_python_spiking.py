"""Validate the in-memory Python neural backend; no CARLsim build or subprocess."""
import argparse
import json
import hashlib
from math import sqrt
from pathlib import Path
from time import perf_counter

import numpy as np

from mstd.calibration import stimulus_frames,write_stimulus,file_sha256
from mstd.neural import SpikeConfig
from mstd.projection import Projection,load_projection
from mstd.simulation import simulate
from mstd.spiking_mt import PythonSpikingMT
from mstd.v1 import ArrayOps
from mstd.v1_recording import dump_json


def controls(backend,device):
    weights=np.zeros((28,8));weights[0,0]=1;weights[1,0]=-1
    projection=Projection('controlled',tuple(map(tuple,weights)),(1.,1.,1.),tuple(np.arange(8)*np.pi/4),1.)
    op=ArrayOps(backend,device)
    trials={}
    for name,drive,inhibition in (('zero',False,1.),('on',True,1.),('off',True,0.),('repeat',True,1.)):
        mt=PythonSpikingMT(op,projection,2,2,SpikeConfig(inhibition_weight=inhibition))
        rates=np.zeros(mt.input_shape,np.float32)
        if drive:rates[0,:2,0,0]=100
        mt.set_rates(op.array(rates))
        totals={k:0. for k in ('v1','exc','inh')}
        exc=np.zeros(mt.shape,np.float32)
        event_hash=hashlib.sha256()
        for _ in range(2000):
            spikes=mt.tick()
            for k in totals:totals[k]+=float(spikes[k].sum())
            exc+=op.host(spikes['exc'])
            event_hash.update(op.host(spikes['v1']).astype('<f4',copy=False).tobytes())
        trials[name]=dict(counts=totals,spatial=exc.tolist(),voltage=op.host(mt.exc.v).tolist(),v1_event_sha256=event_hash.hexdigest())
    on,off=trials['on']['counts'],trials['off']['counts']
    spatial=np.asarray(trials['on']['spatial'])
    checks=dict(zero_input_no_spikes=all(x==0 for x in trials['zero']['counts'].values()),
                poisson_count_screen=abs(on['v1']-400)<5*sqrt(400),
                identical_input_for_ablation=trials['on']['v1_event_sha256']==trials['off']['v1_event_sha256'],
                exc_and_inh_fire=off['exc']>0 and on['inh']>0,
                inhibition_suppresses_exc=on['exc']<off['exc'],
                same_seed_reproducible=trials['on']==trials['repeat'],
                no_spatial_crosstalk=bool(np.all(spatial[:,:,1]==0) and np.all(spatial[:,1,:]==0)))
    return dict(checks=checks,counts={k:v['counts'] for k,v in trials.items()},passed=all(checks.values()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--backend',choices=['numpy','torch'],default='numpy')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--all-directions',action='store_true')
    parser.add_argument('--record-spikes',action='store_true',help='also save per-tick events for image trials')
    args=parser.parse_args()
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=False)
    project=Path(__file__).resolve().parents[1]
    source_hashes={path:file_sha256(project/path) for path in (
        'src/mstd/neural.py','src/mstd/spiking_mt.py','src/mstd/model.py','src/mstd/tensor_pipeline.py',
        'src/mstd/monitors.py','src/mstd/simulation.py','src/mstd/v1.py','configs/v1_projection.json')}
    dump_json(root/'source_hashes.json',source_hashes)
    started=perf_counter()
    print('[controls] zero input / E-I ablation / reproducibility',flush=True)
    control=controls(args.backend,args.device)
    dump_json(root/'controls.json',control)
    projection_path=Path(__file__).resolve().parents[1]/'configs/v1_projection.json'
    projection=load_projection(projection_path)
    rows=[]
    for deg in (range(0,360,45) if args.all_directions else [0]):
        stimulus=root/f'grating_d{deg:03d}.dat'
        write_stimulus(stimulus,32,32,stimulus_frames(32,32,40,deg,1.5))
        output=root/f'd{deg:03d}'
        print(f'[grating] {deg}deg',flush=True)
        report=simulate(dict(mt_backend='python_spiking',mt_radius=0,
                             backend=args.backend,device=args.device,
                             projection_config=str(projection_path),input=dict(path=str(stimulus)),
                             output=str(output),progress_every=10,
                             monitors=dict(spikes=args.record_spikes,neural_state=True)))
        values=np.fromfile(output/'monitors/mt_exc_rates/values.f32',dtype='<f4').reshape(40,8,32,32)
        profile=values[20:,:,4:-4,4:-4].mean(axis=(0,2,3),dtype=np.float64)
        winner=int(profile.argmax());angle=float(np.degrees(projection.directions_rad[winner])%360)
        error=abs((angle-deg+180)%360-180)
        row=dict(input_deg=deg,winner_index=winner,winner_deg=angle,peak_error_deg=error,
                 active=bool(profile.max()>0),direction_passed=bool(profile.max()>0 and error<=22.5),
                 mean_direction_rates_hz=profile.tolist(),initialization_seconds=report['initialization_seconds'],
                 compute_seconds=report['compute_seconds'],wall_seconds=report['wall_seconds'])
        rows.append(row);dump_json(root/'directions.json',rows)
    result=dict(schema='mstd-python-spiking-validation-v1',backend=args.backend,device=args.device,
                controls=control,directions=rows,source_hashes=source_hashes,
                passed=control['passed'] and all(r['direction_passed'] for r in rows),
                wall_seconds=perf_counter()-started,
                limitations=['one seed/speed for image trials; not heading accuracy or biological validation',
                             'continuous MSTd adapter runs each tick; no spiking MSTd implemented yet',
                             'fast conductance model and Poisson count input are not CARLsim numerical replicas'])
    dump_json(root/'validation.json',result)
    print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
