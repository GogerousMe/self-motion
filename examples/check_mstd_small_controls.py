"""Frozen-parameter ideal/noise controls and saved-MT frame-end replay.

Replay holds each saved frame-end vector for 50ms: diagnostic, not exact replay
of original within-frame 1ms MT traces. Original outputs are analyzed separately.
"""
import argparse,csv,hashlib,json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from mstd.data import V1Spec
from mstd.mstd import DynamicsConfig,horizontal_centers
from mstd.tensor_pipeline import TensorMSTd
from mstd.v1 import ArrayOps

PARENTS=['20261109_165103__cloud_static_80x80_f000080__py_spike_cuda__r01',
         '20261109_165321__cloud_forward_80x80_f000080__py_spike_cuda__r01']

def summarize(rows,expected):
    stable=rows[20:];valid=[r for r in stable if r['valid']]
    return dict(evaluation_frames=len(stable),valid_fraction=len(valid)/len(stable),
        max_score_mean=float(np.mean([r['max_score'] for r in stable])),
        max_activity_mean=float(np.mean([r['max_activity'] for r in stable])),
        margin_mean=float(np.mean([r['margin'] for r in stable])),
        evidence_mean=float(np.mean([r['evidence'] for r in stable])),
        instantaneous_mae=float(np.mean([abs((r['instant_angle']-expected+180)%360-180) for r in stable])) if expected is not None else None,
        activity_winner_mae=float(np.mean([abs((r['winner_angle']-expected+180)%360-180) for r in stable])) if expected is not None else None,
        valid_mae=float(np.mean([abs((r['winner_angle']-expected+180)%360-180) for r in valid])) if expected is not None and valid else None)

def row_from_result(case,index,result,angles):
    scores=np.asarray(result['template_scores']);activity=np.asarray(result['activity'])
    order=np.argsort(scores)
    return dict(case=case,frame=index,end_s=(index+1)*.05,valid=bool(result['bearing_valid']),
        max_score=float(scores.max()),score_gap=float(scores[order[-1]]-scores[order[-2]]),
        max_activity=float(activity.max()),margin=result['margin'],evidence=result['mt_evidence'],
        instant_angle=float(angles[scores.argmax()]),winner_angle=float(angles[result['winner_index']]))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];out=Path(a.output)
    if not out.resolve().is_relative_to(root/'experiments/validation'):p.error('use experiments/validation')
    out.mkdir(parents=True,exist_ok=False)
    angles=np.arange(-20,21,5);op=ArrayOps();spec=V1Spec(80,80,tuple(i*np.pi/4 for i in range(8)),40.,40.,39.5,39.5)
    cfg=DynamicsConfig(max_step_s=.001);all_rows=[];summary=[];fingerprints={};examples={}
    def execute(name,frames,expected,definition):
        model=TensorMSTd(op,spec,horizontal_centers(spec,angles),cfg);rows=[]
        for i,vectors in enumerate(frames):
            vectors=np.asarray(vectors,dtype=np.float32)
            if i==0:examples[name]=vectors.copy()
            evidence=float(np.sqrt((vectors*vectors).sum(axis=0)).mean())
            result=model.step(dict(vectors=vectors,evidence=evidence),.05)
            rows.append(row_from_result(name,i,result,angles))
        all_rows.extend(rows);summary.append(dict(case=name,expected=expected,definition=definition,**summarize(rows,expected)))
    y,x=np.indices((80,80));cx,cy=39.5,39.5
    for heading in (0,-10,10,7):
        theta=np.arctan2(y-cy,x-(cx+40*np.tan(np.radians(heading)))).astype(np.float32)
        v=.5*np.stack([np.cos(theta),np.sin(theta)])
        execute(f'ideal_h{heading}'.replace('-','minus'),(v for _ in range(80)),heading,'Constant radial direction field, strength0.5')
    theta=np.arctan2(y-cy,x-cx).astype(np.float32)
    for strength in (.05,.005,.0005,0.):
        v=strength*np.stack([np.cos(theta),np.sin(theta)])
        execute('strength_'+str(strength).replace('.','p'),(v for _ in range(80)),0 if strength else None,
                f'Same radial geometry scaled to {strength}; normalized template scores need not shrink')
    for seed in (11,22,33):
        for sigma in (15,45,90):
            rng=np.random.default_rng(seed)
            def frames():
                for _ in range(80):
                    t=theta+rng.normal(0,np.radians(sigma),(80,80))
                    yield .5*np.stack([np.cos(t),np.sin(t)])
            execute(f'noise_{sigma}_seed_{seed}',frames(),0,f'Per-pixel independent angular Gaussian noise {sigma}deg; strength0.5')
        rng=np.random.default_rng(seed)
        def random_frames():
            for _ in range(80):
                t=rng.uniform(-np.pi,np.pi,(80,80));yield .5*np.stack([np.cos(t),np.sin(t)])
        execute(f'random_background_seed_{seed}',random_frames(),None,'No radial structure; local vector amplitude0.5')
    for parent in PARENTS:
        run=root/'experiments/validation'/parent
        report=json.loads((run/'model_report.json').read_text(encoding='utf-8'))
        expected=0 if 'cloud_forward' in parent else None
        originals=[json.loads(l) for l in (run/'estimates.jsonl').open(encoding='utf-8')]
        assert report['complete'] and len(originals)==80 and report['model']['intrinsics_px']==dict(fx_px=40.00000000000001,fy_px=40.00000000000001,cx_px=39.5,cy_px=39.5)
        directory=run/'monitors/mt_readout_activity';meta=json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
        assert meta['complete'] and meta['sample_indices']==list(range(80)) and meta['shape']==[80,8,80,80]
        path=directory/'values.f32';assert path.stat().st_size==80*8*80*80*4
        fingerprints[parent]=dict(activity_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            estimates_sha256=hashlib.sha256((run/'estimates.jsonl').read_bytes()).hexdigest(),input_sha256=report['input']['stimulus_sha256'])
        values=np.fromfile(path,dtype='<f4').reshape(80,8,80,80);assert np.isfinite(values).all() and (values>=0).all()
        units=np.asarray(meta['directions_rad']);vectors=np.stack([np.einsum('tdyx,d->tyx',values,np.cos(units)),np.einsum('tdyx,d->tyx',values,np.sin(units))],axis=1)
        name='real_forward' if expected is not None else 'real_static'
        rows=[row_from_result(name+'_original',i,e,angles) for i,e in enumerate(originals)]
        all_rows.extend(rows);summary.append(dict(case=name+'_original',expected=expected,definition='Actual saved CUDA outputs, unchanged',**summarize(rows,expected)))
        execute(name+'_zoh',vectors,expected,'Frame-end snapshots held over each 50ms interval; approximate readout replay')
        rng=np.random.default_rng(42);permutation=rng.permutation(6400)
        shuffled=vectors.reshape(80,2,-1)[:,:,permutation].reshape(80,2,80,80)
        execute(name+'_spatial_shuffle',shuffled,None,'Fixed pixel permutation: same local vector values, destroyed spatial geometry')
        # The instantaneous diagnostic score must match original frame-end scores.
        model=TensorMSTd(op,spec,horizontal_centers(spec,angles),cfg)
        residual=0.
        for i,v in enumerate(vectors):
            e=float(np.hypot(v[0],v[1]).mean());r=model.step(dict(vectors=v,evidence=e),.05)
            residual=max(residual,float(np.max(np.abs(np.array(r['template_scores'])-originals[i]['template_scores']))))
        fingerprints[parent]['frame_end_score_max_residual']=residual
        assert residual<2e-6
    with (out/'frames.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    source_paths=('examples/check_mstd_small_controls.py','src/mstd/tensor_pipeline.py','src/mstd/mstd.py')
    result=dict(dynamics=asdict(cfg),summary=summary,parent_checks=fingerprints,
        source_hashes={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in source_paths},
        limitations=['No production parameters changed or gate fitted.','Ideal inputs are numerical direction fields, not image DATs.',
        'Synthetic noise seeds are not independent real scenes.','ZOH replays are not exact within-frame neural replays; use original rows for original coverage.'])
    (out/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    np.savez_compressed(out/'input_first_frames.npz',**examples)
    artifact=dict(schema='mstd-artifact-v1',id=out.name,kind='validation',status='completed',stimulus_ids=[],
        input_paths=['experiments/validation/'+p for p in PARENTS],parent_run_ids=PARENTS,
        notes='Frozen MSTd diagnostic controls and saved readout replay. Numerical fixtures, not image stimuli; no production gate change.')
    (out/'artifact.json').write_text(json.dumps(artifact,indent=2),encoding='utf-8')
    registry=root/'experiments/registry.csv'
    with registry.open(encoding='utf-8-sig',newline='') as f:fields=next(csv.reader(f))
    with registry.open('a',encoding='utf-8',newline='') as f:
        csv.DictWriter(f,fieldnames=fields).writerow(dict(id=out.name,kind='validation',path=out.as_posix(),status='completed',parent_run_ids=';'.join(PARENTS),notes=artifact['notes']))
    for r in summary:print(json.dumps(r))

if __name__=='__main__':main()
