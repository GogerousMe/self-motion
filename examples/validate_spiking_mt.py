"""Real-engine smoke tests with controlled rates, not a heading accuracy benchmark."""
import argparse
import json
from pathlib import Path

import numpy as np

from mstd.calibration import file_sha256,write_stimulus,stimulus_frames
from mstd.simulation import simulate
from mstd.v1_recording import RawWriter,dump_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--mode',choices=['cpu','gpu'],default='cpu')
    parser.add_argument('--grating',action='store_true',help='also run 32x32 rightward grating through real V1-MT-MSTd')
    args=parser.parse_args()
    root=Path(args.output).resolve();root.mkdir(parents=True,exist_ok=False)
    projection=json.loads((Path(__file__).resolve().parents[1]/'configs/v1_projection.json').read_text())
    weights=np.zeros((28,8));weights[0,0]=1;weights[1,0]=-1
    projection['projections']={'controlled':{'weights':weights.tolist()}}
    projection['source']='controlled positive/negative local channel test, not historical weights'
    projection.pop('source_sha256',None)
    projection_path=root/'controlled_projection.json';dump_json(projection_path,projection)
    for name,drive in (('zero',False),('drive',True)):
        writer=RawWriter(root/name,2,2,.05,{'source':'controlled-rate-test'})
        rates=np.zeros((3,28,2,2),np.float32)
        if drive:rates[0,:2,0,0]=100
        for _ in range(40):writer.append(rates)
        writer.close()
    reports={}
    for name,source,inhibition in (('zero','zero',True),('on','drive',True),('off','drive',False),('repeat','drive',True)):
        reports[name]=simulate(dict(mt_backend='carlsim',mt_radius=0,projection_config=str(projection_path),
                                   projection='controlled',input=dict(kind='v1_recording',path=str(root/source)),
                                   layers=dict(v1=False),output=str(root/(name+'_run')),frame_ms=50,progress_every=0,
                                   carlsim=dict(executable=str(Path(args.executable).resolve()),mode=args.mode,
                                                seed=42,inhibition_enabled=inhibition)))
    on,off=reports['on']['event_counts'],reports['off']['event_counts']
    same=all(file_sha256(root/'on_run/spikes'/f'{g}.dat')==
             file_sha256(root/'repeat_run/spikes'/f'{g}.dat')
             for g in ('v1','exc','inh'))
    exc=np.fromfile(root/'on_run/monitors/mt_exc_rates/values.f32',dtype='<f4').reshape(40,8,2,2)
    expected=400 # two driven input neurons, 100Hz, two seconds
    checks=dict(zero_input_no_events=all(v==0 for v in reports['zero']['event_counts'].values()),
                input_count_within_5_sigma=bool(abs(on['v1']-expected)<=5*np.sqrt(expected)),
                inhibitory_cells_fire=on['inh']>0,excitatory_cells_fire=off['exc']>0,
                inhibition_reduces_exc=on['exc']<off['exc'],same_seed_exact_events=same,
                no_spatial_crosstalk=bool(np.all(exc[:,:,1,:]==0) and np.all(exc[:,:,0,1]==0)))
    result=dict(scope='real CARLsim controlled-rate smoke test; no historical equivalence/heading claim',
                checks=checks,event_counts={k:v['event_counts'] for k,v in reports.items()},
                passed=all(checks.values()))
    if args.grating:
        stimulus=root/'grating32.dat'
        write_stimulus(stimulus,32,32,stimulus_frames(32,32,40,0,1.5))
        report=simulate(dict(mt_backend='carlsim',mt_radius=0,input=dict(path=str(stimulus)),
                             projection_config=str(Path(__file__).resolve().parents[1]/'configs/v1_projection.json'),
                             output=str(root/'grating_run'),progress_every=10,
                             carlsim=dict(executable=str(Path(args.executable).resolve()),mode=args.mode)))
        values=np.fromfile(root/'grating_run/monitors/mt_exc_rates/values.f32',dtype='<f4').reshape(40,8,32,32)
        profile=values[20:,:,4:-4,4:-4].mean(axis=(0,2,3),dtype=np.float64)
        result['grating']=dict(native=report['native'],event_counts=report['event_counts'],
                               steady_roi_direction_rates_hz=profile.tolist(),winner_index=int(profile.argmax()),
                               note='one direction one seed; not eight-direction validation or heading accuracy')
    dump_json(root/'validation.json',result)
    print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
