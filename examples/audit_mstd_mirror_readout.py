"""Audit a saved mirror batch; compare native winner and offline local readout.

Local quadratic interpolation is exploratory only. It never rewrites native
estimates, validity decisions, thresholds, or the recorded acceptance result.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mstd.mstd import DynamicsConfig
from mstd.validity import ValidityConfig
from check_mstd_validity import saved_states
from inspect_mstd_cloud_holdout import refined_peak

FREEZE='experiments/validation/20261109__mstd_geometry_gate_80x80_f000080__py_rate_cpu__r01/frozen_validity.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(predicted,target):
    x=np.asarray(predicted,dtype=np.float64)
    error=x-target
    return dict(mean_heading_deg=float(x.mean()),mean_signed_error_deg=float(error.mean()),
                mae_deg=float(np.abs(error).mean()),p95_error_deg=float(np.percentile(np.abs(error),95)),
                max_error_deg=float(np.abs(error).max()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--frozen-config',default=FREEZE)
    p.add_argument('--regression-summary',help='optional existing seven-case metrics.json for exploratory known-data regression')
    a=p.parse_args()
    batch=(ROOT/a.batch).resolve();out=(ROOT/a.output).resolve()
    if not batch.is_relative_to(ROOT/'experiments/validation') or not out.is_relative_to(ROOT/'outputs/comparisons'):
        p.error('use experiments/validation batch and outputs/comparisons output')
    if out.exists():raise FileExistsError(out)
    meta=json.loads((batch/'batch.json').read_text(encoding='utf-8'))
    summary=json.loads((batch/'summary.json').read_text(encoding='utf-8'))
    batch_art=json.loads((batch/'artifact.json').read_text(encoding='utf-8'))
    frozen_path=ROOT/a.frozen_config
    frozen=json.loads(frozen_path.read_text(encoding='utf-8'))
    gate=ValidityConfig(**frozen['validity']);dynamics=DynamicsConfig(**frozen['dynamics'])
    assert batch_art['status']=='completed' and len(meta['runs'])==6
    assert batch_art['validity_freeze_sha256']==digest(frozen_path)
    folder=ROOT/meta['suite']
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    images={r['name']:np.frombuffer((folder/r['stimulus']).read_bytes()[25:],np.uint8).reshape(80,80,80) for r in manifest['runs']}
    assert np.array_equal(images['cloud_heading_left7'],images['cloud_heading_right7'][:,:,::-1])
    original_layers=json.loads((ROOT/batch_art['layer_analysis_path']/'metrics.json').read_text(encoding='utf-8'))
    cases=[];checks={};predictions={};frames=[]
    angles=np.arange(-20,21,5)
    for item in meta['runs']:
        run=ROOT/item['output'];parent=item['run_id']
        definition,rows,check=saved_states(ROOT,parent,gate,dynamics,digest(frozen_path))
        check['native_gate_recomputed_all_80_frames']=True
        config=json.loads((run/'model_config.json').read_text(encoding='utf-8'))
        report=json.loads((run/'model_report.json').read_text(encoding='utf-8'))
        artifact=json.loads((run/'artifact.json').read_text(encoding='utf-8'))
        assert config['spiking']['seed']==item['seed'] and config['device']=='cuda'
        assert config['mstd_validity']==asdict(gate)
        assert report['input']['stimulus_sha256']==batch_art['input_sha256'][meta['suite']+'/'+item['case']+'.dat']
        assert artifact['validity_freeze_sha256']==digest(frozen_path)
        assert json.loads((run/'frozen_validity.json').read_text(encoding='utf-8'))==frozen
        state=[json.loads(s) for s in (run/'estimates.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(state)==80 and all(e['frame']==i and abs(e['estimate_timestamp_s']-(i+1)*.05)<1e-8 for i,e in enumerate(state))
        for monitor,key in (('mstd_activity','activity'),('mstd_scores','template_scores')):
            directory=run/'monitors'/monitor
            info=json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
            assert info['complete'] and info['shape']==[80,9] and info['sample_indices']==list(range(80))
            values=np.fromfile(directory/'values.f32',dtype='<f4')
            assert values.size==80*9 and np.isfinite(values).all()
            np.testing.assert_allclose(values.reshape(80,9),[e[key] for e in state],rtol=0,atol=1e-7)
        stable=state[20:];target=definition['heading']
        original=np.array([angles[e['winner_index']] for e in stable],dtype=float)
        alternate=[];fallback=0
        for index,e in enumerate(stable,20):
            q=refined_peak(np.array(e['activity'],dtype=np.float64),angles)
            if q is None:q=float(angles[e['winner_index']]);fallback+=1
            alternate.append(q)
            frames.append(dict(run_id=parent,seed=item['seed'],case=item['case'],frame=index,
                native_heading_deg=float(angles[e['winner_index']]),offline_activity_interpolation_deg=q,
                expected_heading_deg=target,bearing_valid=e['bearing_valid']))
        alt=np.array(alternate)
        expected=next(r for r in summary['cases'] if r['run_id']==parent)
        original_stats=stats(original,target)
        assert abs(original_stats['mae_deg']-expected['mae_deg'])<1e-9
        assert sum(e['bearing_valid'] for e in stable)/60==expected['valid_fraction']
        cases.append(dict(run_id=parent,seed=item['seed'],case=item['case'],target=target,
            native=original_stats,offline_activity_interpolation=stats(alt,target),
            native_target_passed=expected['diagnostic_passed'],valid_fraction=expected['valid_fraction'],
            native_winner_histogram=dict(Counter(map(float,original))),interpolation_fallback_frames=fallback))
        predictions[(item['seed'],item['case'])]=(original,alt)
        checks[parent]=check
    paired=[]
    for seed in meta['neural_seeds']:
        l=predictions[(seed,'cloud_heading_left7')];r=predictions[(seed,'cloud_heading_right7')]
        paired.append(dict(seed=seed,native_common_bias_deg=float((l[0]+r[0]).mean()/2),
                           offline_interpolation_common_bias_deg=float((l[1]+r[1]).mean()/2)))
    # Independently recompute the archived V1/MT layer statistics to a new path.
    command=[sys.executable,str(ROOT/'examples/inspect_mstd_cloud_holdout.py'),'--suite',meta['suite'],'--output',str(out)]
    for r in meta['runs']:command+=['--parent-run',r['run_id']]
    child=subprocess.run(command,cwd=ROOT,check=True,capture_output=True,text=True)
    (out/'layer_recompute.log').write_text(child.stdout+child.stderr,encoding='utf-8')
    layers=json.loads((out/'metrics.json').read_text(encoding='utf-8'))
    assert layers['parent_checks']==original_layers['parent_checks']
    by_parent={r['parent_run']:r for r in layers['summary']}
    for r in original_layers['summary']:
        new=by_parent[r['parent_run']]
        for key in r['stages']:
            np.testing.assert_allclose(new['stages'][key]['mean_profile'],r['stages'][key]['mean_profile'],atol=1e-10,rtol=0)
    stages=[]
    for name in ('v1_linear','mt_exc_frame_rate','mt_readout'):
        left=[r['stages'][name]['mean_profile_parabolic_peak_diagnostic'] for r in layers['summary'] if r['case']=='cloud_heading_left7']
        right=[r['stages'][name]['mean_profile_parabolic_peak_diagnostic'] for r in layers['summary'] if r['case']=='cloud_heading_right7']
        stages.append(dict(stage=name,left_mean_profile_vertex_deg=float(np.mean(left)),
                           right_mean_profile_vertex_deg=float(np.mean(right)),
                           common_profile_vertex_shift_deg=float((np.mean(left)+np.mean(right))/2)))
    # V1 precedes sampling and should be identical across neural seeds here.
    for case in ('cloud_heading_left7','cloud_heading_right7'):
        hashes={layers['parent_checks'][r['run_id']]['raw_v1_sha256'] for r in meta['runs'] if r['case']==case}
        assert len(hashes)==1
    regression=[];context_parents=[]
    if a.regression_summary:
        old=json.loads((ROOT/a.regression_summary).read_text(encoding='utf-8'))
        for item in old['summary']:
            parent=item['name'];context_parents.append(parent)
            state=[json.loads(s) for s in (ROOT/'experiments/validation'/parent/'estimates.jsonl').read_text(encoding='utf-8').splitlines()][20:]
            assert len(state)==60
            if item['heading'] is None:
                regression.append(dict(run_id=parent,target=None,native_valid_frames=sum(e['bearing_valid'] for e in state),
                                       note='Readout comparison does not change the gate; no static angle accuracy is defined.'))
                continue
            native=np.array([angles[e['winner_index']] for e in state],dtype=float)
            alternate=[]
            for e in state:
                q=refined_peak(np.asarray(e['activity']),angles)
                alternate.append(q if q is not None else float(angles[e['winner_index']]))
            regression.append(dict(run_id=parent,target=item['heading'],native=stats(native,item['heading']),
                                   offline_activity_interpolation=stats(alternate,item['heading'])))
    result=dict(cases=cases,paired_bias=paired,layer_profile_diagnostics=stages,parent_checks=checks,
        earlier_seven_regression=regression,regression_summary_path=a.regression_summary,
        native_passed_conditions=sum(r['native_target_passed'] for r in cases),native_all_passed=summary['all_six_targets_passed'],
        mean_native_mae_deg=float(np.mean([r['native']['mae_deg'] for r in cases])),
        mean_offline_interpolation_mae_deg=float(np.mean([r['offline_activity_interpolation']['mae_deg'] for r in cases])),
        original_batch=batch.relative_to(ROOT).as_posix(),frozen_sha256=digest(frozen_path),
        audit_source_sha256=digest(Path(__file__)),
        limitations=['Interpolation chosen after inspecting these recordings; not independent acceptance.',
                     'No neural/model replay or native estimate modification; alternate values are offline diagnostics only.',
                     'Single scene seed211 and three neural seeds do not prove universal fixed bias.',
                     'Stage profile vertices differ from framewise mean predictions; linear V1 is not an equivalent E/I model.'])
    (out/'audit_metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    import csv
    with (out/'readout_comparison.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(frames[0]));w.writeheader();w.writerows(frames)
    artifact_path=out/'artifact.json';artifact=json.loads(artifact_path.read_text(encoding='utf-8'))
    artifact.update(context_batch_id=batch.name,audit_source_sha256=result['audit_source_sha256'],
                    notes='Independent local saved-layer and gate audit; native acceptance preserved; exploratory readout comparison only.')
    if context_parents:
        artifact['parent_run_ids']=list(dict.fromkeys(artifact['parent_run_ids']+context_parents))
        artifact['input_paths']=['experiments/validation/'+s for s in artifact['parent_run_ids']]
        artifact['context_validation_path']=str(a.regression_summary)
    artifact_path.write_text(json.dumps(artifact,indent=2),encoding='utf-8')
    if context_parents:
        import io
        registry=ROOT/'experiments/registry.csv'
        with registry.open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f);fields=reader.fieldnames;entry=next(r for r in reader if r['id']==out.name)
        entry['parent_run_ids']=';'.join(artifact['parent_run_ids'])
        entry['notes']=artifact['notes']
        lines=registry.read_text(encoding='utf-8').splitlines(keepends=True)
        indices=[i for i,line in enumerate(lines) if line.startswith(out.name+',')];assert len(indices)==1
        replacement=io.StringIO(newline='');csv.DictWriter(replacement,fieldnames=fields).writerow(entry)
        lines[indices[0]]=replacement.getvalue()
        with registry.open('w',encoding='utf-8',newline='') as f:f.write(''.join(lines))
    for r in cases:print(json.dumps(dict(seed=r['seed'],target=r['target'],native_mae=r['native']['mae_deg'],offline_mae=r['offline_activity_interpolation']['mae_deg'])))
    print(json.dumps(dict(native_passed=result['native_passed_conditions'],native_all_passed=result['native_all_passed'],
                          mean_native_mae=result['mean_native_mae_deg'],mean_offline_mae=result['mean_offline_interpolation_mae_deg'],stages=stages)))


if __name__=='__main__':main()
