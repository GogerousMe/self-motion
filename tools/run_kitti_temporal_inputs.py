"""Validate copied KITTI bundles and run the existing temporal comparison entry."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mstd.calibration import file_sha256, inspect_stimulus
from mstd.model import load_model_config, model_config

CLIPS = {'kitti01': (665, 744), 'kitti08': (2681, 2760), 'kitti04': (153, 232)}


def bundle_paths(case):
    start, end = CLIPS[case]
    sid = f'{case}_f{start:06d}_f{end:06d}__160x48__f000080__dt104ms__gray_resize__v01'
    folder = ROOT / 'input/stimuli/natural/kitti' / sid
    return sid, folder, folder / (sid + '.dat')


def register_bundle(sid, folder, dat, meta):
    path = dat.relative_to(ROOT).as_posix()
    entries = [
        ('input/registry.csv', 'path', path,
         dict(path=path, width=160, height=48, frames=80, channels=1,
              sha256=meta['stimulus_sha256'], scope='reusable')),
        ('experiments/registry.csv', 'id', sid,
         dict(id=sid, kind='stimulus', path=folder.relative_to(ROOT).as_posix(),
              status='completed', stimulus_ids=sid, notes='Verified copied KITTI temporal input bundle.'))]
    for registry_name, key, value, row in entries:
        registry = ROOT / registry_name
        with registry.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            fields, known = reader.fieldnames, list(reader)
        matches = [entry for entry in known if entry[key] == value]
        if len(matches) > 1:
            raise ValueError(f'duplicate indexed input {value}')
        if matches:
            if any(str(matches[0][name]) != str(row[name]) for name in ('path', 'sha256') if name in row):
                raise ValueError(f'index disagrees with copied input {value}')
            continue
        with registry.open('a', encoding='utf-8', newline='') as stream:
            csv.DictWriter(stream, fieldnames=fields).writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', nargs='+', choices=CLIPS, default=['kitti01', 'kitti08'])
    parser.add_argument('--config', default='configs/model_python_spiking_temporal_baseline32_v01.json')
    parser.add_argument('--backend', choices=['numpy', 'torch'], default='torch')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    if len(set(args.cases)) != len(args.cases):
        parser.error('each case can appear only once')
    if args.max_frames is not None and not 1 <= args.max_frames <= 80:
        parser.error('--max-frames must be in 1..80')
    config_path = (ROOT / args.config).resolve()
    base = load_model_config(config_path)
    entry = ROOT / 'tools/run_mstd_temporal_end_to_end.py'
    if not entry.is_file():
        raise FileNotFoundError(entry)
    plans, bundles = [], []
    for case in args.cases:
        sid, folder, dat = bundle_paths(case)
        meta = json.loads(dat.with_suffix('.dat.json').read_text(encoding='utf-8-sig'))
        info = inspect_stimulus(dat)
        digest = file_sha256(dat)
        if (info != dict(width=160, height=48, frames=80, channels=1, type=4)
                or meta['stimulus_id'] != sid or meta['frame_ms'] != 104
                or meta['stimulus_sha256'] != digest):
            raise ValueError(f'invalid geometry, timing or checksum for {case}')
        intrinsics_path = folder / meta['intrinsics_file']
        intrinsics = json.loads(intrinsics_path.read_text(encoding='utf-8-sig'))
        if intrinsics != meta['intrinsics_px']:
            raise ValueError(f'intrinsics disagree for {case}')
        truth = [json.loads(line) for line in (folder / meta['truth_file']).read_text(encoding='utf-8').splitlines()]
        if len(truth) != 80 or any(row['frame'] != i or abs(row['input_time_s'] - i * .104) > 1e-8
                                  for i, row in enumerate(truth)):
            raise ValueError(f'truth/frame mapping disagrees for {case}')
        verification_path = folder / 'verification.json'
        if verification_path.is_file():
            verification = json.loads(verification_path.read_text(encoding='utf-8'))
            for name, expected in verification.get('files_sha256', {}).items():
                path = (folder / name).resolve()
                if not path.is_relative_to(folder.resolve()) or file_sha256(path) != expected:
                    raise ValueError(f'incomplete or changed bundle file {name}')
        settings = dict(base, mt_backend='python_spiking', mt_radius=0, frame_ms=104,
                        backend=args.backend, device=args.device, intrinsics=intrinsics)
        settings['spiking'] = dict(base['spiking'], substeps=32, seed=43)
        model_config(settings)
        command = [sys.executable, '-u', str(entry), '--config', str(config_path),
                   '--stimulus', dat.relative_to(ROOT).as_posix(), '--frame-ms', '104',
                   '--intrinsics', intrinsics_path.relative_to(ROOT).as_posix(),
                   '--backend', args.backend, '--device', args.device]
        if args.max_frames is not None:
            command.extend(['--max-frames', str(args.max_frames)])
        plans.append(dict(case=case, stimulus_id=sid, stimulus_sha256=digest, command=command))
        bundles.append((sid, folder, dat, meta))
    device_tag = 'cuda' if args.device.startswith('cuda') else 'cpu'
    plan = dict(cases=plans, runs_per_case=12, total_runs=12 * len(plans),
                mt_branches=['baseline', 'spatial_only'],
                mstd_variants=['reference', 'input_only', 'no_self', 'no_lateral', 'tau40', 'tau160'],
                frame_ms=104, frames_per_run=args.max_frames or 80,
                seed=43, substeps=32, output_root='experiments/validation',
                output_pattern=f'<beijing_time>__mstd_temporal_image__py_spike_{device_tag}__s0043__r01',
                note='Each existing entry invocation prints OUTPUT and writes its stimulus path in plan.json.')
    print(json.dumps(plan, indent=2), flush=True)
    if args.plan_only:
        return
    for bundle in bundles:
        register_bundle(*bundle)
    for case in plans:
        print(f"[suite] starting {case['case']}", flush=True)
        subprocess.run(case['command'], cwd=ROOT, check=True)
        print(f"[suite] completed {case['case']}", flush=True)


if __name__ == '__main__':
    main()
