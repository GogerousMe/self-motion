"""Audit the received KITTI pilot and record task-level heading metrics."""
import argparse
from collections import Counter
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from mstd.calibration import inspect_stimulus
from mstd.data import V1Spec
from mstd.mstd import DynamicsConfig
from mstd.readout import ReadoutConfig, heading_readout
from mstd.tensor_pipeline import TensorMSTd
from mstd.v1 import ArrayOps
from mstd.validity import ValidityConfig, decide_validity


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run = (ROOT/args.recording).resolve()
    output = (ROOT/args.output).resolve()
    if not output.is_relative_to(ROOT/'outputs/comparisons'):
        parser.error('output must be under outputs/comparisons')
    if output.exists():
        raise FileExistsError(output)
    report = load(run/'model_report.json')
    config = load(run/'model_config.json')
    artifact = load(run/'artifact.json')
    states = [json.loads(s) for s in (run/'estimates.jsonl').read_text(encoding='utf-8').splitlines()]
    with (ROOT/'input/registry.csv').open(encoding='utf-8-sig', newline='') as stream:
        matches = [r for r in csv.DictReader(stream) if r['sha256'] == report['input']['stimulus_sha256']]
    assert len(matches) == 1
    stimulus = ROOT/matches[0]['path']
    metadata = load(stimulus.with_suffix('.dat.json'))
    truth = [json.loads(s) for s in (stimulus.parent/metadata['truth_file']).read_text().splitlines()]
    assert sha(stimulus) == metadata['stimulus_sha256'] == artifact['input_sha256']
    assert inspect_stimulus(stimulus) == dict(width=160, height=48, frames=80, channels=1, type=4)
    assert report['complete'] and report['frames'] == len(states) == len(truth) == 80
    assert config['frame_ms'] == metadata['frame_ms'] == 104
    assert report['model']['intrinsics_px'] == metadata['intrinsics_px']
    spec = V1Spec(160, 48, tuple(np.arange(8)*np.pi/4), **metadata['intrinsics_px'])
    centers = report['model']['mstd_template_centers_px']
    readout_config = ReadoutConfig(**config['mstd_readout'])
    gate = ValidityConfig(**config['mstd_validity'])
    templates = TensorMSTd(ArrayOps(), spec, centers).templates.astype(np.float64)
    directory = run/'monitors/mt_readout_activity'
    monitor = load(directory/'metadata.json')
    assert monitor['complete'] and monitor['shape'] == [80,8,48,160] and monitor['sample_indices'] == list(range(80))
    assert (directory/'values.f32').stat().st_size == 80*8*48*160*4
    activity = np.memmap(directory/'values.f32', dtype='<f4', mode='r', shape=(80,8,48,160))
    directions = np.array(monitor['directions_rad'])
    units = np.stack((np.cos(directions), np.sin(directions)), axis=1)
    rows = []
    max_score_residual = 0.
    for i, (state, reference) in enumerate(zip(states, truth)):
        assert state['frame'] == reference['frame'] == i
        assert abs(state['input_timestamp_s']-i*.104) < 1e-8
        assert abs(state['estimate_timestamp_s']-(i+1)*.104) < 1e-8
        assert np.isfinite(activity[i]).all() and (activity[i]>=0).all()
        vectors = np.einsum('dyx,dc->cyx', activity[i], units)
        magnitude = np.hypot(*vectors)
        norm = float(magnitude.sum())
        scores = np.maximum((templates*vectors).sum(axis=(1,2,3))/max(norm, 1e-12), 0)
        max_score_residual = max(max_score_residual, float(np.max(abs(scores-state['template_scores']))))
        uniform = float(np.linalg.norm(vectors.sum(axis=(1,2)))/max(norm,1e-12))
        valid, decision = decide_validity(gate, DynamicsConfig(), state['mt_evidence'], state['margin'],
                                          state['template_scores'], state['winner_index'], uniform,
                                          float((magnitude >= gate.local_evidence_threshold).mean()))
        assert valid == state['valid'] == state['bearing_valid']
        decoded = heading_readout(state['activity'], centers, spec, state['winner_index'], True, readout_config)
        if valid:
            assert abs(decoded['alpha_x_deg']-state['alpha_x_deg']) < 1e-8
        else:
            assert state['alpha_x_deg'] is None
        rows.append(dict(frame=i,source_frame=reference['source_frame'],valid=bool(valid),
            truth=reference['expected_heading_deg'],winner=decoded['winner_alpha_x_deg'],
            selected=decoded['alpha_x_deg'],geometry_excess=decision['geometry_excess'],
            rejected_by=decision['rejected_by']))
    assert max_score_residual < 2e-6
    for name, key in (('mstd_activity','activity'), ('mstd_scores','template_scores')):
        path = run/'monitors'/name/'values.f32'
        values = np.fromfile(path, dtype='<f4')
        assert values.size == 80*9
        np.testing.assert_allclose(values.reshape(80,9), [s[key] for s in states], atol=1e-7, rtol=0)
    activity._mmap.close()
    stable = rows[20:]
    valid_rows = [r for r in stable if r['valid']]
    metrics = dict(valid_frames=len(valid_rows),evaluation_frames=60,valid_fraction=len(valid_rows)/60,
        truth_mean_deg=float(np.mean([r['truth'] for r in stable])),
        always_zero_mae_deg=float(np.mean([abs(r['truth']) for r in stable])),
        selected_mean_deg=float(np.mean([r['selected'] for r in stable])),
        winner_histogram=dict(Counter(r['winner'] for r in stable)),
        rejected_frames=[r['frame'] for r in stable if not r['valid']],
        rejection_reasons=dict(Counter(k for r in stable for k in r['rejected_by'])))
    native = load(run/'heading_metrics.json')
    for key in ('winner','selected'):
        error = np.array([abs(r[key]-r['truth']) for r in stable])
        result = dict(unconditional_mae_deg=float(error.mean()),p95_error_deg=float(np.percentile(error,95)),
            max_error_deg=float(error.max()),valid_mae_deg=float(np.mean([abs(r[key]-r['truth']) for r in valid_rows])))
        for name,value in result.items():
            assert abs(value-native[key][name]) < 1e-8
        metrics[key] = result
    array_inventory = {}
    for path in (run/'monitors').iterdir():
        if path.name in ('v1','spikes') or not path.is_dir():
            continue
        meta_path = path/'metadata.json'
        data_path = path/'values.f32'
        if meta_path.exists():
            meta = load(meta_path)
            array_inventory[path.name] = dict(expected_bytes=int(np.prod(meta['shape']))*4,
                received_bytes=data_path.stat().st_size if data_path.exists() else 0)
    output.mkdir(parents=True, exist_ok=False)
    result = dict(metrics=metrics,score_max_residual=max_score_residual,native_gate_verified_frames=80,
        array_inventory=array_inventory,source_hashes={name:sha(run/name) for name in ('estimates.jsonl','model_config.json','model_report.json')},
        audit_script_sha256=sha(Path(__file__)),notes='Saved-state audit with native104ms input timing and transformed calibration.')
    (output/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (output/'per_frame.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    audit_artifact = dict(schema='mstd-artifact-v1', id=output.name,kind='comparison',status='completed',
        stimulus_ids=artifact['stimulus_ids'],input_paths=[run.relative_to(ROOT).as_posix()],parent_run_ids=[run.name],
        notes='KITTI pilot saved-state audit and project progress review,2026-11-16.')
    (output/'artifact.json').write_text(json.dumps(audit_artifact,indent=2),encoding='utf-8')
    registry = ROOT/'experiments/registry.csv'
    with registry.open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream);fields=reader.fieldnames
        assert not any(r['id']==output.name for r in reader)
    with registry.open('a',encoding='utf-8',newline='') as stream:
        csv.DictWriter(stream,fieldnames=fields).writerow(dict(id=output.name,kind='comparison',path=output.relative_to(ROOT).as_posix(),
            status='completed',parent_run_ids=run.name,stimulus_ids=';'.join(artifact['stimulus_ids']),notes=audit_artifact['notes']))
    print(json.dumps(result))


if __name__=='__main__':main()
