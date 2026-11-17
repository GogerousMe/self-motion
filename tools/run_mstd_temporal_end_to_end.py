"""Two frozen MT branches crossed with six MSTd dynamics settings."""
import argparse
from dataclasses import asdict
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mstd.model import load_model_config
from mstd.simulation import simulate
from mstd.temporal import temporal_variants
from mstd.calibration import file_sha256
from mstd.v1_recording import dump_json
from run_mt_spatial_controls import register_run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--stimulus',required=True)
    parser.add_argument('--frame-ms',type=float,required=True)
    parser.add_argument('--intrinsics',type=Path)
    parser.add_argument('--backend',choices=['numpy','torch'],default='torch')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--max-frames',type=int)
    parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args()
    base=load_model_config(args.config)
    base.update(mt_backend='python_spiking',mt_radius=0,backend=args.backend,device=args.device,
                frame_ms=args.frame_ms,max_frames=args.max_frames)
    base['spiking'].update(substeps=32,seed=43)
    base['mt_competition']['enabled']=False
    if args.intrinsics:base['intrinsics']=json.loads(args.intrinsics.read_text(encoding='utf-8-sig'))
    variants=temporal_variants()
    plan=dict(mt_branches=['baseline','spatial_only'],dynamics={k:asdict(v) for k,v in variants.items()},
              stimulus=args.stimulus,frame_ms=args.frame_ms,max_frames=args.max_frames,
              backend=args.backend,device=args.device,seed=43,substeps=32,command=[sys.executable,*sys.argv])
    if args.plan_only:print(json.dumps(plan,indent=2));return
    stimulus=Path(args.stimulus).resolve()
    stimulus_hash=file_sha256(stimulus)
    stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    device='cuda' if args.device.startswith('cuda') else 'cpu'
    rid=f'{stamp}__mstd_temporal_image__py_spike_{device}__s0043__r01'
    out=ROOT/'experiments/validation'/rid;out.mkdir(parents=True,exist_ok=False)
    dump_json(out/'plan.json',plan)
    artifact=dict(schema='mstd-artifact-v1',id=rid,kind='validation',status='running',parent_run_ids=[],stimulus_ids=[],
        input_paths=[str(stimulus)],input_sha256=stimulus_hash,
        source_hashes={p:file_sha256(ROOT/p) for p in ['src/mstd/model.py','src/mstd/temporal.py','src/mstd/tensor_pipeline.py',
            'src/mstd/mstd.py','src/mstd/spiking_mt.py','src/mstd/mt_circuit.py','tools/run_mstd_temporal_end_to_end.py']},
        notes='Fixed MT baseline/spatial crossed with six MSTd dynamics settings. All neural ticks advanced.')
    hashes={};summaries=[]
    try:
        for branch in plan['mt_branches']:
            for name,dynamics in variants.items():
                settings=json.loads(json.dumps(base))
                first=branch=='baseline' and name=='reference'
                replay=out/'baseline/reference/monitors/v1'
                settings.update(input=dict(kind='stimulus_dat' if first else 'v1_recording',path=str(stimulus if first else replay)),
                    output=str(out/branch/name),mstd_dynamics=asdict(dynamics))
                settings['layers'].update(v1=first,mt=True,mstd=True)
                settings['mt_spatial']['enabled']=branch=='spatial_only'
                settings['monitors'].update(v1=first,mt_normalization=True,mt_projection=True,mstd=True,spikes=False,neural_state=False,every_n_frames=1)
                report=simulate(settings)
                folder=out/branch/name
                digest=file_sha256(folder/'monitors/mt_readout_activity/values.f32')
                if branch not in hashes:hashes[branch]=digest
                if hashes[branch]!=digest:raise AssertionError('MT activity changed across dynamics ablations')
                register_run(folder,dict(**{k:v for k,v in artifact.items() if k not in ('id','status','parent_run_ids')},
                    id=rid+'__'+branch+'__'+name,status='completed',parent_run_ids=[rid]))
                summaries.append(dict(mt_branch=branch,variant=name,complete=report['complete'],frames=report['frames'],mt_activity_sha256=digest))
                dump_json(out/'summary.json',summaries)
        artifact.update(status='completed',mt_frozen_hashes=hashes)
    except BaseException:
        artifact['status']='failed';raise
    finally:register_run(out,artifact)
    print('OUTPUT',out.relative_to(ROOT).as_posix(),flush=True)


if __name__=='__main__':main()
