"""Paired baseline/spatial-only controls through the normal simulation entry."""
import argparse
import csv
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from mstd.calibration import file_sha256, stimulus_frames, write_stimulus
from mstd.execution import execution_settings
from mstd.model import model_config
from mstd.mt_circuit import SpatialConfig, CompetitionConfig
from mstd.neural import SpikeConfig
from mstd.project_paths import resolve_stimulus
from mstd.simulation import simulate
from mstd.v1_recording import dump_json, raw_info


KITTI_RUN = '20261109_234047__kitti04_f000153_f000232_160x48_f000080__py_spike_cuda__s0043__r01'
STATIC_RUN = '20261116_145631__kitti04_static193_160x48_f000080__py_spike_cuda__s0043__r01'
VARIANT_SWITCHES = {
    'baseline': (False, False),
    'spatial_only': (True, False),
    'competition_only': (False, True),
    'spatial_competition': (True, True),
}


def append_index(path, row, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    if exists:
        with path.open(encoding='utf-8-sig', newline='') as stream:
            fields = csv.DictReader(stream).fieldnames
    with path.open('a', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='raise')
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def register_run(folder, artifact):
    dump_json(folder / 'artifact.json', artifact)
    append_index(ROOT / 'experiments/registry.csv', dict(
        id=artifact['id'], kind=artifact['kind'], path=folder.relative_to(ROOT).as_posix(),
        status=artifact['status'], parent_run_ids=';'.join(artifact['parent_run_ids']),
        notes=artifact['notes']),
        ['id', 'kind', 'path', 'legacy_path', 'status', 'parent_run_ids', 'stimulus_ids', 'notes'])


def summarize(folder, case, warmup):
    report = json.loads((folder / 'model_report.json').read_text(encoding='utf-8'))
    model = report['model']
    shape = (report['frames'], 8, model['height'], model['width'])
    frames = shape[0]
    if not 0 <= warmup < frames:
        raise ValueError('warmup must leave evaluation frames')
    rates = np.memmap(folder / 'monitors/mt_exc_rates/values.f32', mode='r', dtype='<f4', shape=shape)
    activity = np.memmap(folder / 'monitors/mt_readout_activity/values.f32', mode='r', dtype='<f4', shape=shape)
    angles = np.asarray(json.loads((folder / 'monitors/mt_exc_rates/metadata.json').read_text())['directions_rad'])
    units = np.stack((np.cos(angles), np.sin(angles)), axis=1)
    border = min(6, (min(shape[2:]) - 1) // 2)
    region = (slice(border, shape[2] - border), slice(border, shape[3] - border))
    global_errors, local_errors, coverage = [], [], []
    try:
        for frame in range(warmup, frames):
            vectors = np.einsum('dyx,dc->cyx', activity[frame], units)[:, region[0], region[1]]
            amplitude = np.linalg.norm(vectors, axis=0)
            active = amplitude >= 0.02
            coverage.append(float(active.mean()))
            if case.get('direction_deg') is not None:
                pooled = vectors.mean(axis=(1, 2))
                if np.linalg.norm(pooled) > 1e-12:
                    angle = np.degrees(np.arctan2(pooled[1], pooled[0]))
                    global_errors.append(float(abs((angle - case['direction_deg'] + 180) % 360 - 180)))
                local = np.degrees(np.arctan2(vectors[1], vectors[0]))
                local_errors.extend(abs((local[active] - case['direction_deg'] + 180) % 360 - 180).tolist())
        estimates = [json.loads(line) for line in (folder / 'estimates.jsonl').read_text().splitlines()][warmup:]
        result = dict(frames=frames, warmup_frames=warmup, evaluation_frames=frames - warmup,
                      mt_mean_hz=float(rates[warmup:].mean()), mt_max_frame_hz=float(rates[warmup:].max()),
                      active_fraction=float(np.mean(coverage)), mstd_valid_frames=sum(s['valid'] for s in estimates),
                      bearing_valid_frames=sum(s['bearing_valid'] for s in estimates), border_px=border,
                      direction_evaluable_frames=len(global_errors),
                      global_direction_mae_deg=float(np.mean(global_errors)) if global_errors else None,
                      active_local_direction_mae_deg=float(np.mean(local_errors)) if local_errors else None,
                      active_local_direction_p95_deg=float(np.percentile(local_errors, 95)) if local_errors else None,
                      wall_seconds=report['wall_seconds'])
        competition_file = folder / 'monitors/mt_competition_rates/values.f32'
        if competition_file.is_file():
            j_rates = np.memmap(competition_file, mode='r', dtype='<f4', shape=shape)
            try:
                result['competition_mean_hz'] = float(j_rates[warmup:].mean())
                result['competition_max_frame_hz'] = float(j_rates[warmup:].max())
            finally:
                j_rates._mmap.close()
        truth = case.get('truth')
        if truth is not None:
            errors = [abs((s['alpha_x_deg'] - t['expected_heading_deg'] + 180) % 360 - 180)
                      for s, t in zip(estimates, truth[warmup:frames])
                      if s['bearing_valid'] and t.get('expected_heading_deg') is not None]
            result['valid_heading_mae_deg'] = float(np.mean(errors)) if errors else None
        return result
    finally:
        rates._mmap.close()
        activity._mmap.close()


def build_cases(args):
    if args.quick_check:
        return [dict(name='static_texture', static=True), dict(name='texture_d000_s1', direction_deg=0, speed=1.0)]
    cases = []
    if 'static' in args.cases:
        cases.append(dict(name='static_texture', static=True))
    if 'texture' in args.cases:
        cases.extend(dict(name=f'texture_d{angle:03d}_s{speed:g}'.replace('.', 'p'), direction_deg=angle, speed=speed)
                     for speed in (0.5, 1.0, 1.5, 2.0) for angle in range(0, 360, 45))
    if 'kitti' in args.cases:
        for name, parent in (('kitti_static', STATIC_RUN), ('kitti_dynamic', KITTI_RUN)):
            folder = ROOT / 'experiments/validation' / parent
            meta = raw_info(folder / 'monitors/v1')
            report = json.loads((folder / 'model_report.json').read_text())
            if not report['complete'] or meta.get('monitor_every_n_frames', 1) != 1:
                raise ValueError(f'complete every-frame V1 required for {parent}')
            case = dict(name=name, parent=parent, recording=folder / 'monitors/v1',
                        frames=meta['frames'], frame_ms=meta['frame_interval_s'] * 1000,
                        intrinsics=report['model']['intrinsics_px'], static=name == 'kitti_static')
            if name == 'kitti_dynamic':
                expected_hash = report['input']['stimulus_sha256']
                stimulus = resolve_stimulus(report['input']['input_path'], expected_hash, ROOT)
                if file_sha256(stimulus) != expected_hash:
                    raise ValueError('KITTI stimulus hash differs from the parent run')
                sidecar = json.loads(stimulus.with_suffix('.dat.json').read_text(encoding='utf-8-sig'))
                truth_path = stimulus.parent / sidecar['truth_file']
                case['truth'] = [json.loads(line) for line in truth_path.read_text().splitlines()]
                if len(case['truth']) < case['frames']:
                    raise ValueError('KITTI truth does not cover every input frame')
                case['truth_sha256'] = file_sha256(truth_path)
            cases.append(case)
    return cases


def run(args, variants=('baseline', 'spatial_only')):
    if (not variants or variants[0] != 'baseline' or len(set(variants)) != len(variants)
            or any(variant not in VARIANT_SWITCHES for variant in variants)):
        raise ValueError('unique supported variants must start with baseline for V1 replay')
    competition_suite = any(VARIANT_SWITCHES[variant][1] for variant in variants)
    execution = execution_settings(args.backend, args.device)
    spatial = SpatialConfig(enabled=True, gain=args.gain, radius=args.radius)
    competition = CompetitionConfig(
        enabled=True,
        input_weight=getattr(args, 'competition_input_weight', 0.25),
        inhibition_weight=getattr(args, 'competition_weight', 0.2),
        mode=getattr(args, 'competition_mode', 'angular'))
    neural = replace(SpikeConfig(), seed=args.seed, substeps=args.substeps)
    cases = build_cases(args)
    plan = dict(schema='mstd-mt-spatial-controls-v1', execution=execution, quick_check=args.quick_check,
                command=[sys.executable, *sys.argv],
                spiking=neural.description(), mt_spatial=spatial.description(),
                cases=[{k: str(v) if isinstance(v, Path) else v for k, v in case.items() if k != 'truth'} for case in cases],
                warmup_frames=20, texture_width=32 if args.quick_check else 80,
                texture_frames=24 if args.quick_check else 40, stimulus_seed=0,
                comparison='baseline versus spatial-only; same V1 replay, seed and neural integration',
                numerical_scope='finite-state checks at every integration substep; saved voltages/conductances at frame end')
    if competition_suite:
        plan.update(schema='mstd-mt-competition-controls-v1', variants=list(variants),
                    mt_competition=competition.description(),
                    comparison='four-way spatial/competition ablation with shared V1 replay, seed and integration')
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    device_tag = 'cuda' if execution['device'].startswith('cuda') else 'cpu'
    suite_name = 'mt_competition_controls' if competition_suite else 'mt_spatial_controls'
    run_id = f'{stamp}__{suite_name}__py_spike_{device_tag}__sub{args.substeps:02d}__s{args.seed:04d}__r01'
    output = ROOT / 'experiments/validation' / run_id
    output.mkdir(parents=True, exist_ok=False)
    dump_json(output / 'plan.json', plan)
    sources = ['src/mstd/mt_circuit.py', 'src/mstd/spiking_mt.py', 'src/mstd/neural.py',
               'src/mstd/model.py', 'src/mstd/monitors.py', 'src/mstd/v1.py', 'src/mstd/tensor_pipeline.py',
               'src/mstd/simulation.py', 'src/mstd/mstd.py', 'src/mstd/validity.py', 'src/mstd/readout.py',
               'configs/v1_projection.json', 'tools/run_mt_spatial_controls.py']
    if competition_suite:
        sources.append('tools/run_mt_competition_controls.py')
    artifact = dict(schema='mstd-artifact-v1', id=run_id, kind='validation', status='running',
                    parent_run_ids=[c['parent'] for c in cases if 'parent' in c], stimulus_ids=[], input_paths=[],
                    source_hashes={path: file_sha256(ROOT / path) for path in sources},
                    notes='Spatial-only pilot, gain is provisional; direction competition and heading feedback disabled.')
    if competition_suite:
        artifact['notes'] = 'Four-way spatial/competition pilot with provisional parameters.'
    dump_json(output / 'artifact.json', artifact)
    results = []
    print('OUTPUT', output.relative_to(ROOT).as_posix(), flush=True)
    try:
        for case in cases:
            name = case['name']
            folder = output / name
            folder.mkdir()
            source = case.get('recording')
            source_kind = 'v1_recording' if source else 'stimulus_dat'
            if source is None:
                width = plan['texture_width']
                payload = list(stimulus_frames(width, width, plan['texture_frames'], case.get('direction_deg', 0),
                                               case.get('speed', 1), kind='texture', seed=plan['stimulus_seed']))
                if case.get('static'):
                    payload = [payload[0]] * len(payload)
                source = folder / 'stimulus.dat'
                write_stimulus(source, width, width, payload)
                source_hash = file_sha256(source)
                dump_json(source.with_suffix('.dat.json'), dict(schema='mstd-spatial-fixture-v1', width=width,
                    height=width, frames=len(payload), frame_ms=50, sha256=source_hash,
                    coordinates='x-right_y-down; degrees clockwise from right', **case))
                append_index(ROOT / 'input/registry.csv', dict(path=source.relative_to(ROOT).as_posix(),
                    width=width, height=width, frames=len(payload), channels=1, sha256=source_hash, scope='fixture'),
                    ['path', 'width', 'height', 'frames', 'channels', 'sha256', 'scope'])
            else:
                source_hash = file_sha256(source / 'rates.f32')
            artifact['input_paths'].append(source.relative_to(ROOT).as_posix())
            dump_json(folder / 'case.json', {k: str(v) if isinstance(v, Path) else v for k, v in case.items()})
            pair = {}
            for variant in variants:
                spatial_enabled, competition_enabled = VARIANT_SWITCHES[variant]
                destination = folder / variant
                replay = variant != 'baseline' or source_kind == 'v1_recording'
                input_path = folder / 'baseline/monitors/v1' if variant != 'baseline' else source
                settings = model_config(dict(**execution, mt_backend='python_spiking', mt_radius=0,
                    projection_config=str(ROOT / 'configs/v1_projection.json'),
                    frame_ms=case.get('frame_ms', 50), spiking=neural.description(),
                    mt_spatial=replace(spatial, enabled=spatial_enabled).description(),
                    mt_competition=replace(competition, enabled=competition_enabled).description(),
                    intrinsics=case.get('intrinsics'), input=dict(kind='v1_recording' if replay else 'stimulus_dat', path=str(input_path)),
                    output=str(destination), layers=dict(v1=not replay),
                    mstd_validity=dict(mode='geometry_support'), mstd_readout=dict(mode='local_quadratic'),
                    monitors=dict(v1=variant == 'baseline', spikes=False, neural_state=True), progress_every=10))
                child_id = run_id + '__' + name + '__' + variant
                parents = [run_id + '__' + name + '__baseline'] if variant != 'baseline' else ([case['parent']] if 'parent' in case else [])
                child = dict(schema='mstd-artifact-v1', id=child_id, kind='validation', status='running',
                    parent_run_ids=parents, stimulus_ids=[], input_paths=[input_path.relative_to(ROOT).as_posix()],
                    input_sha256=file_sha256(input_path / 'rates.f32') if replay else source_hash,
                    source_hashes=artifact['source_hashes'], notes=artifact['notes'])
                try:
                    simulate(settings)
                    child['status'] = 'completed'
                finally:
                    if child['status'] != 'completed':
                        child['status'] = 'failed'
                    if destination.exists():
                        register_run(destination, child)
                pair[variant] = summarize(destination, case, plan['warmup_frames'])
                results.append(dict(case=name, variant=variant, **pair[variant]))
            if 'spatial_only' in pair:
                pair['mt_mean_hz_change_pct'] = 100 * (pair['spatial_only']['mt_mean_hz'] / max(pair['baseline']['mt_mean_hz'], 1e-12) - 1)
            if competition_suite:
                pair['variant_rate_change_pct'] = {
                    variant: 100 * (pair[variant]['mt_mean_hz'] / max(pair['baseline']['mt_mean_hz'], 1e-12) - 1)
                    for variant in variants}
            pair['static_input'] = bool(case.get('static'))
            dump_json(folder / 'comparison.json', pair)
            dump_json(output / 'summary.json', results)
            print('COMPARED', name, json.dumps(pair), flush=True)
        artifact['status'] = 'completed'
    except BaseException:
        artifact['status'] = 'failed'
        raise
    finally:
        register_run(output, artifact)


def parse_args(argv=None, competition=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['numpy', 'torch'])
    parser.add_argument('--device')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--quick-check', action='store_true')
    parser.add_argument('--cases', default='static,texture', help='comma-separated static,texture,kitti')
    parser.add_argument('--substeps', type=int, default=32)
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--gain', type=float, default=0.05)
    parser.add_argument('--radius', type=int, default=2)
    if competition:
        parser.description = 'Four-way MT spatial integration and spiking direction competition controls.'
        parser.add_argument('--competition-input-weight', type=float, default=0.25)
        parser.add_argument('--competition-weight', type=float, default=0.2)
        parser.add_argument('--competition-mode', choices=['angular', 'opponent'], default='angular')
    args = parser.parse_args(argv)
    args.cases = args.cases.split(',')
    if not args.cases or any(case not in ('static', 'texture', 'kitti') for case in args.cases):
        parser.error('unsupported --cases')
    return args


def main():
    run(parse_args())


if __name__ == '__main__':
    main()
