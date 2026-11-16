"""Apply both formal readouts to saved activities, without neural replay."""
import argparse,csv,hashlib,json
from dataclasses import asdict
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mstd.data import V1Spec
from mstd.readout import ReadoutConfig,heading_readout


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);a=p.parse_args()
    out=(ROOT/a.output).resolve()
    if not out.is_relative_to(ROOT/'experiments/validation'):p.error('use experiments/validation')
    out.mkdir(parents=True,exist_ok=False)
    batch=ROOT/'experiments/validation/20261109_222324__mirror7_s0211_80x80_f000480__py_spike_cuda__r01'
    mirror=json.loads((batch/'summary.json').read_text())['cases']
    old=json.loads((ROOT/'experiments/validation/20261109__mstd_gate_seven_cases_80x80_f000080__py_rate_cpu__r01/metrics.json').read_text())['summary']
    cases=[(r['run_id'],r['expected_heading_deg']) for r in mirror]+[(r['name'],r['heading']) for r in old]
    previous=json.loads((ROOT/'outputs/comparisons/selfmotion/20261109__mirror7_s0211__readout_audit__v02/audit_metrics.json').read_text())
    expected={r['run_id']:r['offline_activity_interpolation']['mae_deg'] for r in previous['cases']}
    expected.update({r['run_id']:r['offline_activity_interpolation']['mae_deg'] for r in previous['earlier_seven_regression'] if r['target'] is not None})
    summary=[];hashes={};rows=[];cfg=ReadoutConfig(mode='local_quadratic')
    for parent,target in cases:
        run=ROOT/'experiments/validation'/parent
        report=json.loads((run/'model_report.json').read_text());states=[json.loads(s) for s in (run/'estimates.jsonl').read_text().splitlines()]
        assert report['complete'] and len(states)==report['frames']==80
        spec=V1Spec(report['model']['width'],report['model']['height'],tuple(np.arange(8)*np.pi/4),**report['model']['intrinsics_px'])
        centers=report['model']['mstd_template_centers_px'];grid=[];local=[];valid=[];fallback=0
        for i,s in enumerate(states):
            winner=heading_readout(s['activity'],centers,spec,s['winner_index'],True)
            candidate=heading_readout(s['activity'],centers,spec,s['winner_index'],True,cfg)
            published=heading_readout(s['activity'],centers,spec,s['winner_index'],s['valid'],cfg)
            if not s['valid']:assert published['alpha_x_deg'] is None
            rows.append(dict(run_id=parent,frame=i,winner_alpha_x_deg=winner['alpha_x_deg'],
                candidate_alpha_x_deg=candidate['alpha_x_deg'],published_candidate_alpha_x_deg=published['alpha_x_deg'],
                bearing_valid=published['bearing_valid'],target=target))
            if i>=20:
                grid.append(winner['alpha_x_deg']);local.append(candidate['alpha_x_deg']);valid.append(s['bearing_valid'])
                fallback+=candidate['readout']['method_used']!='local_quadratic'
        angles=[np.degrees(np.arctan((c[0]-spec.cx_px)/spec.fx_px)) for c in centers]
        row=dict(run_id=parent,target=target,group='static' if target is None else 'grid_center' if min(abs(np.array(angles)-target))<1e-8 else 'off_grid',
                 valid_fraction=float(np.mean(valid)),fallback_frames=fallback)
        for label,values in (('winner',grid),('local_quadratic',local)):
            row[label+'_mae_deg']=float(np.mean(np.abs(np.array(values)-target))) if target is not None else None
        if target is not None:assert abs(row['local_quadratic_mae_deg']-expected[parent])<1e-9
        summary.append(row);hashes[parent]=sha(run/'estimates.jsonl')
    with (out/'readouts.jsonl').open('x',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
    result=dict(summary=summary,readout_config=asdict(cfg),parent_estimate_hashes=hashes,
        source_hashes={p:sha(ROOT/p) for p in ('src/mstd/readout.py','src/mstd/model.py','src/mstd/tensor_pipeline.py','examples/check_mstd_readout_recordings.py')},
        limitation='Known-data formal implementation regression. Not new held-out acceptance; no neural trajectory replay.')
    (out/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (out/'readout_frozen.json').write_text(json.dumps(dict(schema='mstd-readout-freeze-v1',config=asdict(cfg),
        scope='ascending uniform horizontal angular grid only',status='candidate, known-data regression completed; independent acceptance pending'),indent=2),encoding='utf-8')
    artifact=dict(schema='mstd-artifact-v1',id=out.name,kind='validation',status='completed',stimulus_ids=[],
        input_paths=['experiments/validation/'+parent for parent,target in cases],parent_run_ids=[p for p,t in cases],notes=result['limitation'])
    (out/'artifact.json').write_text(json.dumps(artifact,indent=2),encoding='utf-8')
    registry=ROOT/'experiments/registry.csv'
    with registry.open(encoding='utf-8-sig',newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;assert not any(r['id']==out.name for r in reader)
    with registry.open('a',encoding='utf-8',newline='') as f:csv.DictWriter(f,fieldnames=fields).writerow(dict(id=out.name,kind='validation',path=out.relative_to(ROOT).as_posix(),status='completed',parent_run_ids=';'.join(artifact['parent_run_ids']),notes=artifact['notes']))
    print(json.dumps(dict(recordings=len(summary),frames=len(rows),regression_matches_exploratory=True,output=str(out))))


if __name__=='__main__':main()
