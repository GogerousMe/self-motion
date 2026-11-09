"""Inspect saved V1/MT/template stages without changing the frozen decoder.

Linear V1 readout is a diagnostic comparator, not an equivalent E/I network.
Parabolic profile peaks below are diagnostics only, never replacement outputs.
"""
import argparse
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from mstd.data import V1Spec
from mstd.mstd import horizontal_centers
from mstd.projection import load_projection
from mstd.tensor_pipeline import DirectionMT, TensorMSTd
from mstd.v1 import ArrayOps
from mstd.v1_recording import raw_info


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1024*1024), b''):
            result.update(part)
    return result.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def array_monitor(run, name):
    folder = run/'monitors'/name
    meta = json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    assert meta['complete'] and meta['shape'] == [80,8,80,80]
    assert meta['sample_indices'] == list(range(80))
    assert (folder/'values.f32').stat().st_size == 80*8*80*80*4
    return np.memmap(folder/'values.f32', mode='r', dtype='<f4', shape=(80,8,80,80))


def profiles(v, patterns):
    norm = float(np.hypot(v[0], v[1]).sum())
    return np.maximum(np.einsum('kcyx,cyx->k', patterns, v, optimize=True)/max(norm,1e-12), 0)


def refined_peak(scores, angles):
    k = int(np.argmax(scores))
    if k == 0 or k == len(angles)-1:
        return None
    left, mid, right = scores[k-1:k+2]
    curvature = left-2*mid+right
    if curvature >= -1e-12:
        return None
    return float(angles[k]+.5*(left-right)/curvature*(angles[1]-angles[0]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-run', action='append', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    out = Path(a.output).resolve()
    if not out.is_relative_to(ROOT/'outputs/comparisons'):
        p.error('use outputs/comparisons')
    out.mkdir(parents=True, exist_ok=False)
    angles = np.arange(-20,21,5)
    op = ArrayOps()
    projection = load_projection(ROOT/'configs/v1_projection.json')
    spec = V1Spec(80,80,projection.directions_rad,40.,40.,39.5,39.5)
    patterns = TensorMSTd(op,spec,horizontal_centers(spec,angles)).templates
    linear = DirectionMT(op,projection,80,80,radius=0)
    bundle = ROOT/'input/stimuli/selfmotion/mstd_gate_holdout_s0097__80x80__f000080__dt50ms__7cases__v01'
    manifest = json.loads((bundle/'manifest.json').read_text(encoding='utf-8'))
    all_rows = []; summaries = []; fingerprints = {}
    for parent in a.parent_run:
        run = ROOT/'experiments/validation'/parent
        report = json.loads((run/'model_report.json').read_text(encoding='utf-8'))
        config = json.loads((run/'model_config.json').read_text(encoding='utf-8'))
        info = next(r for r in manifest['runs'] if r['stimulus_sha256'] == report['input']['stimulus_sha256'])
        expected = info['motion']['heading_deg'] if info['motion']['mode']=='straight' else None
        raw_meta = raw_info(run/'monitors/v1')
        assert raw_meta['frames']==80 and raw_meta['provenance']['stimulus_sha256']==report['input']['stimulus_sha256']
        assert config['mt_radius']==0 and config['projection']=='motionProj1'
        assert report['model']['mstd_template_centers_px'] == [list(c) for c in horizontal_centers(V1Spec(80,80,projection.directions_rad,**manifest['intrinsics_px']),angles)]
        raw = np.memmap(run/'monitors/v1/rates.f32', mode='r', dtype='<f4', shape=(80,3,28,80,80))
        readout = array_monitor(run,'mt_readout_activity')
        exc = array_monitor(run,'mt_exc_rates')
        states = [json.loads(s) for s in (run/'estimates.jsonl').read_text(encoding='utf-8').splitlines()]
        true_pattern = (TensorMSTd(op,spec,[horizontal_centers(spec,[expected])[0],(-100.,-100.)]).templates[0]
                        if expected is not None else None)
        stage_profiles = {s:[] for s in ('v1_linear','mt_exc_frame_rate','mt_readout','ideal_same_amplitude')}
        rows = []; max_residual = 0.
        for i in range(80):
            assert np.isfinite(raw[i]).all() and (raw[i]>=0).all()
            assert np.isfinite(exc[i]).all() and (exc[i]>=0).all()
            assert np.isfinite(readout[i]).all() and (readout[i]>=0).all()
            v1 = linear.step(raw[i])
            vread = np.einsum('dyx,dc->cyx',readout[i],linear.units,optimize=True)
            normalized_exc = exc[i]/(1+exc[i].sum(axis=0))
            vexc = np.einsum('dyx,dc->cyx',normalized_exc,linear.units,optimize=True)
            vectors = dict(v1_linear=v1['vectors'],mt_exc_frame_rate=vexc,mt_readout=vread)
            if true_pattern is not None:
                vectors['ideal_same_amplitude'] = true_pattern*np.hypot(vread[0],vread[1])
            row = dict(case=info['name'],frame=i,legacy_valid=states[i]['validity']['legacy_valid'],
                       geometry_valid=states[i]['valid'],competition_heading=float(angles[states[i]['winner_index']]),
                       raw_v1_mean_hz=float(raw[i].mean(dtype=np.float64)),
                       linear_projection_mean=float(v1['direction_rates'].mean(dtype=np.float64)),
                       mt_exc_mean_hz=float(exc[i].mean(dtype=np.float64)), mt_evidence=states[i]['mt_evidence'])
            for name,v in vectors.items():
                scores = profiles(v,patterns)
                stage_profiles[name].append(scores)
                row[name+'_peak'] = float(angles[scores.argmax()])
                if name=='mt_readout':
                    max_residual = max(max_residual,float(np.max(np.abs(scores-states[i]['template_scores']))))
            if true_pattern is None:
                row['ideal_same_amplitude_peak'] = None
            rows.append(row)
        assert max_residual<2e-6
        stable = rows[20:]
        summary = dict(case=info['name'],parent_run=parent,expected_heading_deg=expected,evaluation_frames=60,
                       legacy_valid_fraction=sum(r['legacy_valid'] for r in stable)/60,
                       geometry_valid_fraction=sum(r['geometry_valid'] for r in stable)/60,
                       winner_histogram=dict(Counter(r['competition_heading'] for r in stable)),
                       instantaneous_competition_disagreements=sum(r['mt_readout_peak']!=r['competition_heading'] for r in stable),
                       original_scores_max_residual=max_residual,
                       v1_first_last_stable_max_difference=float(np.max(np.abs(raw[20]-raw[-1]))))
        for key in ('raw_v1_mean_hz','linear_projection_mean','mt_exc_mean_hz','mt_evidence'):
            summary[key] = float(np.mean([r[key] for r in stable]))
        summary['stages'] = {}
        for name,scores in stage_profiles.items():
            if not scores:
                continue
            avg = np.mean(scores[20:],axis=0)
            peaks = [r[name+'_peak'] for r in stable]
            summary['stages'][name] = dict(mean_profile=avg.tolist(),peak_histogram=dict(Counter(peaks)),
                mean_profile_grid_peak=float(angles[avg.argmax()]),
                mean_profile_parabolic_peak_diagnostic=refined_peak(avg,angles),
                grid_mae_deg=float(np.mean(np.abs(np.asarray(peaks)-expected))) if expected is not None else None)
        summaries.append(summary); all_rows.extend(rows)
        fingerprints[parent] = dict(raw_v1_sha256=sha(run/'monitors/v1/rates.f32'),
                                    estimates_sha256=sha(run/'estimates.jsonl'))
        raw._mmap.close(); readout._mmap.close(); exc._mmap.close()
        print(json.dumps(summary),flush=True)
    dump(out/'metrics.json',dict(summary=summaries,parent_checks=fingerprints,template_angles_deg=angles.tolist(),
         source_sha256=sha(Path(__file__)),projection_sha256=sha(ROOT/'configs/v1_projection.json'),
         limitations=['Linear V1 readout is not a neural E/I counterfactual.',
                      'MT frame-rate readout and filtered trace use different time windows.',
                      'Parabolic peaks are diagnostics, never changed outputs or acceptance tests.',
                      'One scene seed and one neural seed; cannot establish all causal origins.']))
    with (out/'frames.csv').open('x',encoding='utf-8',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
    artifact = dict(schema='mstd-artifact-v1',id=out.name,kind='comparison',status='completed',
                    stimulus_ids=[bundle.name+'__'+s['case'] for s in summaries],parent_run_ids=a.parent_run,
                    input_paths=['experiments/validation/'+s for s in a.parent_run],
                    notes='Read-only saved-stage diagnostics; frozen gate and model unchanged.')
    dump(out/'artifact.json',artifact)
    with (ROOT/'experiments/registry.csv').open(encoding='utf-8-sig',newline='') as f:
        reader = csv.DictReader(f);fields=reader.fieldnames
        assert not any(r['id']==out.name for r in reader)
    with (ROOT/'experiments/registry.csv').open('a',encoding='utf-8',newline='') as f:
        csv.DictWriter(f,fieldnames=fields).writerow(dict(id=out.name,kind='comparison',path=out.relative_to(ROOT).as_posix(),
          status='completed',parent_run_ids=';'.join(a.parent_run),stimulus_ids=';'.join(artifact['stimulus_ids']),notes=artifact['notes']))


if __name__ == '__main__':
    main()
