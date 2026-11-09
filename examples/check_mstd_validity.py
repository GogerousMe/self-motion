"""Develop/freeze a readout gate, then evaluate numerical holdouts and saved MT.

Saved-state checks re-gate ORIGINAL frame-end scores and competition winners.
No zero-order-hold replay and no re-estimation of neural trajectories is used.
"""
import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from mstd.data import V1Spec
from mstd.mstd import DynamicsConfig, horizontal_centers
from mstd.tensor_pipeline import TensorMSTd
from mstd.v1 import ArrayOps
from mstd.validity import ValidityConfig, decide_validity


DEFAULT_PARENTS = [
    '20261109_165103__cloud_static_80x80_f000080__py_spike_cuda__r01',
    '20261109_165321__cloud_forward_80x80_f000080__py_spike_cuda__r01',
]
SPEC = V1Spec(80, 80, tuple(np.arange(8)*np.pi/4), 40., 40., 39.5, 39.5)
ANGLES = np.arange(-20, 21, 5)
Y, X = np.indices((80, 80))
SOURCE_FILES = ('examples/check_mstd_validity.py', 'src/mstd/validity.py',
                'src/mstd/tensor_pipeline.py', 'src/mstd/mstd.py', 'src/mstd/model.py',
                'configs/model_python_spiking_geometry_v01.json')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def radial(heading=0, strength=.5):
    theta = np.arctan2(Y-39.5, X-(39.5+40*np.tan(np.radians(heading))))
    return (strength*np.stack((np.cos(theta), np.sin(theta)))).astype(np.float32)


def case(name, phase, expected, heading=None, kind='radial', seed=0, sigma=0, strength=.5):
    return dict(name=name, phase=phase, expected_valid=expected, heading=heading,
                kind=kind, seed=seed, sigma=sigma, strength=strength)


def fixture_frames(definition):
    d = definition
    rng = np.random.default_rng(d['seed'])
    base = radial(d['heading'] or 0, d['strength'])
    theta = np.arctan2(base[1], base[0])
    for _ in range(80):
        v = base.copy()
        if d['kind'] == 'noise':
            t = theta+rng.normal(0, np.radians(d['sigma']), (80, 80))
            v = d['strength']*np.stack((np.cos(t), np.sin(t)))
        elif d['kind'] == 'random':
            t = rng.uniform(-np.pi, np.pi, (80, 80))
            v = d['strength']*np.stack((np.cos(t), np.sin(t)))
        elif d['kind'] == 'uniform':
            t = np.radians(d['heading'] or 0)
            v = np.broadcast_to(d['strength']*np.array([np.cos(t), np.sin(t)])[:, None, None], (2,80,80)).copy()
        elif d['kind'] == 'patch':
            v[:, 8:, :] = 0
        elif d['kind'] == 'lower_half':
            v[:, :40, :] = 0
        elif d['kind'] == 'rotation':
            v = np.stack((-v[1], v[0]))
        elif d['kind'] == 'contraction':
            v = -v
        yield np.asarray(v, dtype=np.float32)


def make_row(name, index, state, source_valid=None):
    detail = state['validity']
    return dict(case=name, frame=index, end_s=(index+1)*.05,
                source_valid=source_valid, legacy_valid=detail['legacy_valid'],
                geometry_valid=bool(state['valid']),
                winner_heading_deg=float(ANGLES[state['winner_index']]),
                evidence=float(state['mt_evidence']), margin=float(state['margin']),
                **{k:detail[k] for k in ('winner_template_score','max_template_score',
                                        'uniform_score','geometry_excess','active_fraction')},
                rejected_by=';'.join(detail['rejected_by']))


def summarize(definition, rows):
    stable = rows[20:]
    result = dict(**definition, evaluation_frames=len(stable))
    for mode in ('legacy', 'geometry'):
        valid = [r for r in stable if r[mode+'_valid']]
        result[mode+'_valid_fraction'] = len(valid)/len(stable)
        result[mode+'_valid_mae_deg'] = (float(np.mean([abs(r['winner_heading_deg']-definition['heading']) for r in valid]))
                                        if valid and definition['expected_valid'] else None)
    result['unconditional_winner_mae_deg'] = (float(np.mean([abs(r['winner_heading_deg']-definition['heading']) for r in stable]))
                                              if definition['expected_valid'] else None)
    result['mean_score'] = float(np.mean([r['winner_template_score'] for r in stable]))
    result['mean_excess'] = float(np.mean([r['geometry_excess'] for r in stable]))
    result['mean_active_fraction'] = float(np.mean([r['active_fraction'] for r in stable]))
    # Scope-specific diagnostic targets, NOT end-to-end real-video acceptance.
    if definition['expected_valid'] is None:
        result['diagnostic_passed'] = None
    elif definition['expected_valid']:
        result['diagnostic_passed'] = (result['geometry_valid_fraction'] >= .95
                                       and result['unconditional_winner_mae_deg'] <= 2.5+1e-6)
    else:
        result['diagnostic_passed'] = result['geometry_valid_fraction'] <= .05
    return result


def numerical_cases(phase):
    definitions = []
    headings = (0, -10, 10, 7) if phase == 'development' else (-17, -12, -7, -2.5, 2.5, 12, 17)
    seeds = (101,) if phase == 'development' else (303, 404, 505)
    for h in headings:
        definitions.append(case(f'{phase}_clean_{h}', phase, True, h))
    for seed in seeds:
        for sigma in (15, 45, 90):
            h = 0 if phase == 'development' else 12
            definitions.append(case(f'{phase}_noise_{sigma}_s{seed}', phase, True, h, 'noise', seed, sigma))
        definitions.append(case(f'{phase}_random_s{seed}', phase, False, kind='random', seed=seed))
    for h in ((0, 90) if phase == 'development' else (45, 135, 225, 315)):
        definitions.append(case(f'{phase}_uniform_{h}', phase, False, h, 'uniform'))
    for strength, expected in ((.05, True), (.005, False), (0., False)):
        definitions.append(case(f'{phase}_strength_{strength}', phase, expected, 0, strength=strength))
    definitions.extend([case(f'{phase}_patch', phase, False, 0, 'patch'),
                        case(f'{phase}_lower_half', phase, True, 0, 'lower_half')])
    if phase == 'holdout':
        definitions.extend([case('holdout_rotation', phase, False, 0, 'rotation'),
                            case('holdout_contraction', phase, False, 0, 'contraction')])
    return definitions


def evaluate_numerical(definition, gate, dynamics):
    model = TensorMSTd(ArrayOps(), SPEC, horizontal_centers(SPEC, ANGLES), dynamics, gate)
    rows = []
    for i, v in enumerate(fixture_frames(definition)):
        state = model.step(dict(vectors=v, evidence=float(np.hypot(v[0], v[1]).mean())), .05)
        rows.append(make_row(definition['name'], i, state))
    return rows


def saved_states(root, parent, gate, dynamics, freeze_sha256):
    run = root/'experiments/validation'/parent
    report = json.loads((run/'model_report.json').read_text(encoding='utf-8'))
    config = json.loads((run/'model_config.json').read_text(encoding='utf-8'))
    assert report['complete'] and report['frames'] == 80 and config['frame_ms'] == 50
    assert config.get('mstd_template_count', 9) == 9
    assert np.allclose([report['model']['intrinsics_px'][k] for k in ('fx_px','fy_px','cx_px','cy_px')], [40,40,39.5,39.5])
    assert np.allclose(report['model']['mstd_template_centers_px'], horizontal_centers(SPEC, ANGLES))
    with (root/'input/registry.csv').open(encoding='utf-8-sig', newline='') as f:
        indexed = [r for r in csv.DictReader(f) if r['sha256'] == report['input']['stimulus_sha256']]
    if len(indexed) != 1:
        raise ValueError('expected exactly one indexed source stimulus')
    stimulus = root/indexed[0]['path']
    assert digest(stimulus) == indexed[0]['sha256']
    manifest = json.loads((stimulus.parent/'manifest.json').read_text(encoding='utf-8'))
    info = next(r for r in manifest['runs'] if r['stimulus_sha256'] == indexed[0]['sha256'])
    mode = info['motion']['mode']
    expected = True if mode == 'straight' else False if mode == 'static' else None
    heading = info['motion']['heading_deg'] if expected else None
    independent = manifest.get('validity_freeze_sha256') == freeze_sha256
    definition = dict(name=parent, phase='saved_real_independent' if independent else 'saved_real_known', expected_valid=expected, heading=heading,
                      input_sha256=indexed[0]['sha256'])
    states = [json.loads(line) for line in (run/'estimates.jsonl').read_text(encoding='utf-8').splitlines()]
    folder = run/'monitors/mt_readout_activity'
    meta = json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    assert meta['complete'] and meta['shape'] == [80,8,80,80] and meta['sample_indices'] == list(range(80))
    values = np.fromfile(folder/'values.f32', dtype='<f4')
    assert len(states) == 80 and values.size == 80*8*80*80
    values = values.reshape(80,8,80,80)
    assert np.isfinite(values).all() and (values >= 0).all()
    units = np.array(meta['directions_rad'])
    vectors = np.stack((np.einsum('tdyx,d->tyx',values,np.cos(units)),
                        np.einsum('tdyx,d->tyx',values,np.sin(units))), axis=1)
    templates = TensorMSTd(ArrayOps(), SPEC, horizontal_centers(SPEC, ANGLES)).templates
    rows = []; residual = 0.; evidence_residual = 0.
    for i, (v, state) in enumerate(zip(vectors, states)):
        amp = np.hypot(v[0], v[1]); norm = amp.sum()
        scores = np.maximum((templates*v).sum(axis=(1,2,3))/max(float(norm),1e-12),0)
        if norm <= 1e-12:
            scores[:] = 0
        residual = max(residual, float(np.max(np.abs(scores-state['template_scores']))))
        evidence_residual = max(evidence_residual, abs(float(amp.mean())-state['mt_evidence']))
        uniform = float(np.linalg.norm(v.sum(axis=(1,2)))/max(float(norm),1e-12))
        fraction = float((amp >= gate.local_evidence_threshold).mean())
        valid, detail = decide_validity(gate, dynamics, state['mt_evidence'], state['margin'],
                                        state['template_scores'], state['winner_index'], uniform, fraction)
        if config.get('mstd_validity', {}).get('mode') == 'geometry_support':
            assert config['mstd_validity'] == asdict(gate), 'source and frozen gate settings differ'
            assert bool(state['valid']) == valid, 'saved device gate and scalar recomputation disagree'
        else:
            assert bool(state['valid']) == detail['legacy_valid'], 'legacy rule mismatch'
        rows.append(make_row(parent, i, {**state, 'valid':valid, 'validity':detail}, state['valid']))
    assert residual < 2e-6 and evidence_residual < 2e-6
    hashes = dict(mt_sha256=digest(folder/'values.f32'), estimates_sha256=digest(run/'estimates.jsonl'),
                  frame_end_score_max_residual=residual, evidence_max_residual=evidence_residual)
    return definition, rows, hashes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--parent-run', action='append', help='run ID under experiments/validation; repeatable')
    p.add_argument('--frozen-config', type=Path, help='use a previous freeze file; evaluate only supplied/default parents')
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(a.output).resolve()
    if not out.is_relative_to(root/'experiments/validation'):
        p.error('output must be under experiments/validation')
    out.mkdir(parents=True, exist_ok=False)
    parents = a.parent_run or DEFAULT_PARENTS
    all_rows = []; summary = []; checks = {}
    if a.frozen_config:
        frozen = json.loads(a.frozen_config.read_text(encoding='utf-8'))
        gate = ValidityConfig(**frozen['validity'])
        dynamics = DynamicsConfig(**frozen['dynamics'])
        if gate.mode != 'geometry_support' or not frozen['development_passed']:
            p.error('use qualified geometry_support frozen parameters')
        assert frozen['template_angles_deg'] == ANGLES.tolist()
        frozen['reused_from'] = a.frozen_config.as_posix()
        frozen['reused_sha256'] = digest(a.frozen_config)
        write_json(out/'frozen_validity.json', frozen)
        freeze_sha256 = digest(a.frozen_config)
    else:
        gate = ValidityConfig(mode='geometry_support')
        dynamics = DynamicsConfig(max_step_s=.001)
        for d in numerical_cases('development'):
            rows = evaluate_numerical(d, gate, dynamics)
            all_rows.extend(rows); summary.append(summarize(d, rows))
        dev_passed = all(r['diagnostic_passed'] for r in summary)
        frozen = dict(schema='mstd-validity-freeze-v1', validity=asdict(gate), dynamics=asdict(dynamics),
                      template_angles_deg=ANGLES.tolist(), frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                      development_passed=dev_passed, development_cases=[r['name'] for r in summary],
                      selection='Preset informed by prior diagnostics; no holdout/real-response threshold search.',
                      scope='In-view pure-translation expansion; not speed, rotation or SE3 pose.')
        # Freeze on disk BEFORE generating/evaluating held-out random seeds and angles.
        write_json(out/'frozen_validity.json', frozen)
        freeze_sha256 = digest(out/'frozen_validity.json')
        if dev_passed:
            for d in numerical_cases('holdout'):
                rows = evaluate_numerical(d, gate, dynamics)
                all_rows.extend(rows); summary.append(summarize(d, rows))
        else:
            print('Development targets failed; parameters recorded but not qualified for holdouts.', flush=True)
    for parent in parents:
        d, rows, fingerprints = saved_states(root, parent, gate, dynamics, freeze_sha256)
        all_rows.extend(rows); summary.append(summarize(d, rows)); checks[parent] = fingerprints
    metrics = dict(schema='mstd-validity-check-v1', summary=summary, parent_checks=checks,
                   source_hashes={f:digest(root/f) for f in SOURCE_FILES},
                   numerical_holdout_passed=(all(r['diagnostic_passed'] for r in summary if r['phase']=='holdout')
                                               if any(r['phase']=='holdout' for r in summary) else None),
                   independent_image_targets_passed=(all(r['diagnostic_passed'] for r in summary if r['phase']=='saved_real_independent')
                                                      if any(r['phase']=='saved_real_independent' for r in summary) else None),
                   limitations=['Saved real scenes were inspected in prior diagnostics; they are NOT blind holdouts.',
                                'Saved-state checks re-gate original scores/winners exactly; no neural trajectory replay.',
                                'Synthetic noise seeds do not substitute for independent image/neural runs.',
                                'No V1/MT/competition changes; angles remain a 5-degree grid.'])
    write_json(out/'metrics.json', metrics)
    report_lines = ['# MSTd有效性对照', '',
        '同一竞争峰值分别应用旧规则和冻结的新规则。排除前20帧，统计后60帧。分数和活动无量纲，不是Hz或概率。', '',
        '| 条件 | 数据分组 | 旧有效率 | 新有效率 | 未门控峰值MAE(度) | 诊断目标通过 |',
        '|---|---|---:|---:|---:|---|']
    for r in summary:
        mae = '--' if r['unconditional_winner_mae_deg'] is None else f"{r['unconditional_winner_mae_deg']:.3f}"
        report_lines.append(f"| {r['name']} | {r['phase']} | {r['legacy_valid_fraction']:.1%} | {r['geometry_valid_fraction']:.1%} | {mae} | {r['diagnostic_passed']} |")
    report_lines.extend(['', '目标：本轮正例有效率≥95%且未门控峰值MAE≤2.5°；负例误报≤5%。这些是本协议的诊断目标，不是完整自运动系统验收。', '',
        '已有真实静止/前进曾用于问题定位，属于known；只有与当前冻结文件哈希匹配的后生成图像套件记为independent。',
        '保存响应检查直接使用原始分数/竞争赢家，没有重放神经轨迹；几何模式的原生判断还与CPU标量重算逐帧核对。',
        '旋转/后退为当前扩张模板的域外拒绝控制，拒绝不代表已能估计这些运动。', '',
        '参数见frozen_validity.json，逐帧证据见frames.csv，完整指标与来源哈希见metrics.json。'])
    (out/'REPORT.md').write_text('\n'.join(report_lines)+'\n', encoding='utf-8')
    with (out/'frames.csv').open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
    artifact = dict(schema='mstd-artifact-v1', id=out.name, kind='validation', status='completed',
                    stimulus_ids=[], input_paths=['experiments/validation/'+p for p in parents],
                    parent_run_ids=parents, notes='Frozen opt-in geometric gate; known real saved states and numerical holdouts; not end-to-end acceptance.')
    write_json(out/'artifact.json', artifact)
    registry = root/'experiments/registry.csv'
    with registry.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f); fields = reader.fieldnames
        if any(r['id']==out.name for r in reader):
            raise ValueError('duplicate registry ID')
    with registry.open('a', encoding='utf-8', newline='') as f:
        csv.DictWriter(f, fieldnames=fields).writerow(dict(id=out.name, kind='validation', path=out.relative_to(root).as_posix(),
                                                        status='completed', parent_run_ids=';'.join(parents), notes=artifact['notes']))
    for r in summary:
        print(f"{r['name']}: legacy={r['legacy_valid_fraction']:.3f}, geometry={r['geometry_valid_fraction']:.3f}, target={r['diagnostic_passed']}", flush=True)
    print(json.dumps(dict(report=str(out/'metrics.json'), numerical_holdout_passed=metrics['numerical_holdout_passed'])))


if __name__ == '__main__':
    main()
