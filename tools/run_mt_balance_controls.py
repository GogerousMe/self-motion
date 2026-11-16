"""Matched-input MT balance controls and explicit direct-inhibition candidates."""
import argparse,csv,hashlib,json,os,sys
from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime,timedelta,timezone
from pathlib import Path
import numpy as np

ROOT=Path(os.environ.get('MSTD_PROJECT_ROOT',Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(ROOT/'src'))
from mstd.v1 import ArrayOps,MotionEnergyV1
from mstd.spiking_mt import PythonSpikingMT
from mstd.neural import SpikeConfig
from mstd.projection import load_projection
from mstd.tensor_pipeline import TensorMSTd
from mstd.data import V1Spec
from mstd.mstd import horizontal_centers
from mstd.validity import ValidityConfig
from mstd.readout import ReadoutConfig
from mstd.calibration import stimulus_frames,write_stimulus

class DirectInhibition:
    """Use the same delayed V1 weighted events for both E and inhibitory synapses."""
    def __init__(self,mt):self.mt=mt
    def reset(self):pass
    def read(self):return self.mt.input_delay.read()[1]
    def write(self,value):pass

def make_mt(op,projection,width,height,variant):
    config=SpikeConfig(seed=43)
    if variant=='direct_tau5':config=replace(config,tau_inh_ms=5.)
    mt=PythonSpikingMT(op,projection,width,height,config)
    if variant!='baseline':mt.inhib_delay=DirectInhibition(mt)
    return mt

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',default='torch',choices=['torch','numpy'])
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--smoke-only',action='store_true')
    parser.add_argument('--quick-check',action='store_true',help='one32x32 texture,24frames through normal case/output path')
    parser.add_argument('--substeps',type=int,default=4,help='Neuron integration substeps per1ms tick')
    parser.add_argument('--cases',default='static,kitti,texture',help='Comma-separated static,kitti,texture')
    args=parser.parse_args()
    selected_cases=args.cases.split(',')
    if not selected_cases or any(c not in ['static','kitti','texture'] for c in selected_cases):parser.error('invalid --cases')
    replace(SpikeConfig(),substeps=args.substeps)
    variants=['baseline','direct_tau10','direct_tau5']
    plan=dict(variants=variants,seed=43,kitti_frames=80,texture_frames=40,texture_warmup=20,
        texture_speeds=[.5,1.,1.5,2.],texture_directions=list(range(0,360,45)),
        candidate='negative weighted V1 events directly drive MT-E inhibitory conductance; I population still monitored',
        tau_inh_ms=[10,10,5],feedback_enabled=False,quick_check=args.quick_check,substeps=args.substeps,cases=selected_cases)
    if args.plan_only:print(json.dumps(plan,indent=2));return
    op=ArrayOps(args.backend,args.device)
    projection=load_projection(ROOT/'configs/v1_projection.json')
    if args.smoke_only:
        for variant in variants:
            mt=make_mt(op,projection,2,2,variant);mt.config=replace(mt.config,substeps=args.substeps);mt.set_rates(op.zeros((3,28,2,2)))
            events=op.zeros((3,28,2,2))+1
            mt.tick(v1_events=events)
            expected=mt.input_delay.read()[1]
            mt.tick(v1_events=op.zeros((3,28,2,2)))
            if variant!='baseline':assert np.allclose(op.host(mt.gi.g),op.host(expected),atol=1e-6)
            assert np.isfinite(op.host(mt.readout()['activity'])).all()
        print(json.dumps(dict(smoke_passed=True,direct_inhibition_arrival_verified=True,backend=op.backend,device=op.device)));return
    stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    rid=stamp+'__mt_balance_controls__py_spike_'+('cuda' if args.device.startswith('cuda') else 'cpu')+f'__sub{args.substeps:02d}__s0043__r01'
    out=ROOT/'experiments/validation'/rid;out.mkdir(parents=True,exist_ok=False)
    (out/'plan.json').write_text(json.dumps(plan,indent=2))
    parents=['20261116_145631__kitti04_static193_160x48_f000080__py_spike_cuda__s0043__r01',
        '20261109_234047__kitti04_f000153_f000232_160x48_f000080__py_spike_cuda__s0043__r01']
    cases=[dict(name=name,recording=ROOT/'experiments/validation'/parent) for name,parent in zip(['static','kitti'],parents)]
    cases.extend(dict(name=f'texture_d{direction:03d}_s{speed:g}'.replace('.','p'),direction=direction,speed=speed)
        for speed in plan['texture_speeds'] for direction in plan['texture_directions'])
    cases=[c for c in cases if ('texture' if 'direction' in c else c['name']) in selected_cases]
    if args.quick_check:cases=[dict(name='quick_texture_d000_s1',direction=0,speed=1.)]
    artifact=dict(schema='mstd-artifact-v1',id=rid,kind='validation',status='running',parent_run_ids=parents,stimulus_ids=[],
        input_paths=[],notes='Matched V1 inputs, frozen baseline and direct-inhibitory topology/tau controls.',
        source_hashes={'tools/run_mt_balance_controls.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    for source in ['src/mstd/spiking_mt.py','src/mstd/neural.py','src/mstd/v1.py','src/mstd/tensor_pipeline.py','configs/v1_projection.json']:
        artifact['source_hashes'][source]=hashlib.sha256((ROOT/source).read_bytes()).hexdigest()
    summaries=[]
    print('OUTPUT',out.relative_to(ROOT).as_posix(),flush=True)
    try:
        for case in cases:
            folder=out/case['name'];folder.mkdir()
            recording=case.get('recording')
            if recording:
                report=json.loads((recording/'model_report.json').read_text())
                meta=json.loads((recording/'monitors/v1/metadata.json').read_text())
                width,height,count=meta['width'],meta['height'],meta['frames'];dt=meta['frame_interval_s']
                assert (recording/'monitors/v1/rates.f32').stat().st_size==count*3*28*height*width*4
                raw=np.memmap(recording/'monitors/v1/rates.f32',mode='r',dtype='<f4',shape=(count,3,28,height,width))
                camera=report['model']['intrinsics_px'];truth=None
                if case['name']=='kitti':
                    stimulus=ROOT/report['input']['input_path'];sidecar=json.loads(stimulus.with_suffix('.dat.json').read_text())
                    truth=[json.loads(line) for line in (stimulus.parent/sidecar['truth_file']).read_text().splitlines()]
                generator=(op.array(raw[i]) for i in range(count))
                case_config=dict(recording=str(recording.relative_to(ROOT)),v1_sha256=hashlib.sha256((recording/'monitors/v1/rates.f32').read_bytes()).hexdigest(),frame_ms=dt*1000)
            else:
                width=height=32 if args.quick_check else 80
                count=24 if args.quick_check else 40
                dt=.05;camera=None;truth=None
                payload=list(stimulus_frames(width,height,count,case['direction'],case['speed'],kind='texture',seed=0))
                fixture=folder/'stimulus.dat';write_stimulus(fixture,width,height,payload)
                artifact['input_paths'].append(fixture.relative_to(ROOT).as_posix())
                fixture_registry=ROOT/'input/registry.csv'
                with fixture_registry.open(encoding='utf-8-sig',newline='') as stream:
                    fixture_fields=csv.DictReader(stream).fieldnames
                with fixture_registry.open('a',encoding='utf-8',newline='') as stream:
                    csv.DictWriter(stream,fieldnames=fixture_fields).writerow(dict(path=fixture.relative_to(ROOT).as_posix(),width=width,height=height,frames=count,channels=1,sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),scope='fixture'))
                frontend=MotionEnergyV1(width,height,dt,args.backend,args.device)
                generator=(frontend.step(np.frombuffer(bytes(pixels),dtype='u1').reshape(height,width),i*dt) for i,pixels in enumerate(payload))
                case_config=dict(direction_deg=case['direction'],speed_px_per_frame=case['speed'],frames=count,width=width,height=height,frame_ms=50,fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest())
            (folder/'case.json').write_text(json.dumps(case_config,indent=2))
            models={v:make_mt(op,projection,width,height,v) for v in variants}
            for model in models.values():model.config=replace(model.config,substeps=args.substeps)
            heads={}
            if camera:
                spec=V1Spec(width,height,projection.directions_rad,**camera)
                heads={v:TensorMSTd(op,spec,horizontal_centers(spec,np.linspace(-20,20,9)),
                    validity_config=ValidityConfig(mode='geometry_support'),readout_config=ReadoutConfig(mode='local_quadratic')) for v in variants}
            streams={v:(folder/(v+'.jsonl')).open('w') for v in variants}
            binaries={v:(folder/(v+'.activity.f32')).open('wb') for v in variants}
            records={v:[] for v in variants}
            timing=[]
            try:
                with op.torch.inference_mode() if op.torch else nullcontext():
                    for i,raw_rates in enumerate(generator):
                        assert np.isfinite(op.host(raw_rates)).all()
                        for variant,mt in models.items():
                            mt.set_rates(raw_rates);counts=op.zeros((8,height,width));inh_counts=op.zeros((8,height,width))
                            state=None;ticks=round(dt*1000)
                            with op.validation_scope():
                                for tick in range(ticks):
                                    events=mt.tick();counts+=events['exc'];inh_counts+=events['inh']
                                    readout=mt.readout(host_evidence=tick==ticks-1)
                                    if variant in heads:state=heads[variant].step(readout,.001,return_state=tick==ticks-1)
                                    if i in (0,20) and tick<20:
                                        timing.append(dict(frame=i,tick_ms=tick+1,variant=variant,exc_spikes=float(op.host(events['exc']).sum()),
                                            inh_spikes=float(op.host(events['inh']).sum()),ge_mean=float(op.host(mt.ge.g).mean()),gi_mean=float(op.host(mt.gi.g).mean())))
                            activity=op.host(readout['activity']);vector=np.einsum('dyx,dc->cyx',activity,np.asarray(mt.op.host(mt.units)))
                            binaries[variant].write(np.asarray(activity,dtype='<f4').tobytes())
                            pooled=vector[:,6:-6,6:-6].mean(axis=(1,2))
                            direction=float(np.degrees(np.arctan2(pooled[1],pooled[0]))%360)
                            row=dict(frame=i,mt_mean_hz=float(op.host(counts).mean()/dt),inh_mean_hz=float(op.host(inh_counts).mean()/dt),
                                evidence=float(readout['evidence']),direction_deg=direction,resultant=float(np.linalg.norm(pooled)),
                                exc_conductance_mean=float(op.host(mt.ge.g).mean()),inh_conductance_mean=float(op.host(mt.gi.g).mean()),
                                max_mt_hz=float(op.host(counts).max()/dt),min_exc_voltage=float(op.host(mt.exc.v).min()),mstd=state)
                            if 'direction' in case:row['direction_error_deg']=abs((direction-case['direction']+180)%360-180)
                            if truth and state['bearing_valid']:row['heading_error_deg']=abs((state['alpha_x_deg']-truth[i]['expected_heading_deg']+180)%360-180)
                            streams[variant].write(json.dumps(row)+'\n');records[variant].append(row)
                        if i%20==0:print(case['name'],i,'/',count,flush=True)
            finally:
                for stream in streams.values():stream.close()
                for stream in binaries.values():stream.close()
            for variant,mt in models.items():
                (folder/(variant+'.metadata.json')).write_text(json.dumps(dict(complete=True,shape=[count,8,height,width],layout='TDYX',dtype='float32-le',units='dimensionless',frame_interval_s=dt,directions_rad=projection.directions_rad,spiking=mt.config.description(),variant=variant),indent=2))
            (folder/'timing.json').write_text(json.dumps(timing,indent=2))
            for variant,rows in records.items():
                stable=rows[20:];valid=[r for r in stable if r['mstd'] and r['mstd']['bearing_valid']]
                item=dict(case=case['name'],variant=variant,frames=count,mt_mean_hz=float(np.mean([r['mt_mean_hz'] for r in stable])),
                    evidence=float(np.mean([r['evidence'] for r in stable])),valid_frames=len(valid))
                if 'direction' in case:item['direction_mae_deg']=float(np.mean([r['direction_error_deg'] for r in stable]))
                if truth:item['heading_valid_mae_deg']=float(np.mean([r['heading_error_deg'] for r in valid])) if valid else None
                summaries.append(item)
            (out/'summary.json').write_text(json.dumps(summaries,indent=2))
        artifact['status']='completed'
    except BaseException:
        artifact['status']='failed';raise
    finally:
        (out/'artifact.json').write_text(json.dumps(artifact,indent=2))
        registry=ROOT/'experiments/registry.csv'
        with registry.open(encoding='utf-8-sig',newline='') as f:fields=csv.DictReader(f).fieldnames
        with registry.open('a',encoding='utf-8',newline='') as f:csv.DictWriter(f,fieldnames=fields).writerow(dict(id=rid,kind='validation',path=out.relative_to(ROOT).as_posix(),status=artifact['status'],parent_run_ids=';'.join(parents),notes=artifact['notes']))
if __name__=='__main__':main()
