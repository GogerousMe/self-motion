"""One-time, checked, non-overwriting layout migration. Default is dry-run.

Every original file is hashed before moving and checked afterwards. Historical
metadata is never rewritten. See docs/FILE_ORGANIZATION.md before using.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
AUDIT = ROOT / 'docs/migrations/20261107_layout'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def checked(relative):
    p = (ROOT / relative).resolve()
    if not p.is_relative_to(WORKSPACE) or p == WORKSPACE or p == ROOT:
        raise ValueError(f'outside intended project data: {relative}')
    return p


def plan():
    jobs = []

    def add(old, new, kind):
        source, target = checked(old), checked(new)
        if not source.exists():
            raise FileNotFoundError(source)
        if target.exists() or source.is_symlink() or target.is_relative_to(source):
            raise ValueError(f'unsafe/conflicting move: {old} -> {new}')
        jobs.append(dict(old=old, new=new, kind=kind))

    natural = {
        'daria_walk_180x144_gray': ('daria_walk', '180x144', 84, 'gray_native'),
        'daria_walk_188x144_gray_pad': ('daria_walk', '188x144', 84, 'gray_pad'),
        'davis_paragliding_first70_gray': ('davis_paragliding', '854x480', 70, 'gray_native'),
        'snowboard_first65_gray': ('snowboard', '854x480', 65, 'gray_native'),
        'snowboard_gray': ('snowboard', '854x480', 66, 'gray_native'),
    }
    for name, (dataset, grid, frames, variant) in natural.items():
        sid = f'{dataset}__{grid}__f{frames:06d}__dt50ms__{variant}__v01'
        dest = f'input/stimuli/natural/{dataset}/{sid}/{sid}.dat'
        add(f'outputs/stimuli/{name}.dat', dest, 'stimulus')
        add(f'outputs/stimuli/{name}.dat.json', dest + '.json', 'stimulus_metadata')

    bundles = {
        'outputs/calibration_grating_01': 'grating/grating_calibration__80x80__f000080__dt50ms__8dirs__v01',
        'outputs/calibration_texture_01': 'texture/texture_calibration__80x80__f000080__dt50ms__8dirs__v01',
        'outputs/matlab_bw_grating_02': 'grating/bw_grating_suite__80x80__f000080__dt50ms__v02',
        'outputs/selfmotion_01': 'selfmotion/selfmotion_suite__80x80__f000080__dt50ms__14cases__v01',
        'outputs/validation_sequences_01/texture': 'texture/texture_sequence__80x80__f001800__dt50ms__v01',
        'outputs/validation_sequences_01/selfmotion': 'selfmotion/selfmotion_sequence__80x80__f000400__dt50ms__v01',
        'srcipts/bw_grating_output/all_in_one_01': 'grating/grating_sequence__80x80__f004800__dt50ms__v01',
        'srcipts/bw_grating_output/sequence_20261019_134626_053': 'grating/grating_sequence__80x80__f004800__dt50ms__v02',
        'srcipts/bw_grating_output/sequence_20261024_235711_679': 'grating/grating_sequence__80x80__f004800__dt50ms__v03',
    }
    for old, new in bundles.items():
        add(old, 'input/stimuli/' + new, 'stimulus_bundle')

    runs = {
        'daria_walk_gpu_01': 'daria_walk/legacy__daria_walk_180x144_f000084__py_rate_cuda__r01',
        'davis_paragliding_gpu_70_01': 'davis_paragliding/legacy__davis_paragliding_854x480_f000070__py_rate_cuda__r01',
        'snowboard_gpu_65_01': 'snowboard/legacy__snowboard_854x480_f000065__py_rate_cuda__r01',
    }
    for old, new in runs.items():
        add('outputs/' + old, 'outputs/recordings/' + new, 'recording')

    views = {
        'daria_walk_video_01': runs['daria_walk_gpu_01'] + '/mt_stages__v01',
        'python_spiking_visualization_01': 'grating/legacy__grating_32x32_f000040__py_spike_cpu__r03/mt_spikes__v01',
    }
    for old, new in views.items():
        add('outputs/' + old, 'outputs/visualizations/' + new, 'visualization')

    for seq, suffix in [('daria_walk', '01'), ('daria_walk', '02'),
                        ('daria_walk', 'corrected_01'), ('davis_paragliding', '01'), ('snowboard', '01')]:
        variant = {'01': 'v01', '02': 'v02', 'corrected_01': 'v03_corrected'}[suffix]
        add(f'outputs/{seq}_mt_comparison_{suffix}',
            f'outputs/comparisons/{seq}/legacy__{seq}__carlsim_vs_py_rate__{variant}', 'comparison')

    validation = {
        'python_spiking_validation_01': 'legacy__grating_32x32_f000040__py_spike_cpu__r01',
        'python_spiking_validation_final': 'legacy__grating_32x32_f000040__py_spike_cpu__r02',
        'python_spiking_validation_final8': 'legacy__grating_32x32_f000040__py_spike_cpu__r03',
        'calibration_pending_report_01': 'legacy__calibration_pending__r01',
        'grating_mt_run01': 'legacy__grating_80x80_f004800__mt_analysis__r01',
        'grating_mt_run02': 'legacy__grating_80x80_f004800__mt_analysis__r02',
    }
    for n in ['01', '02', '03', '04', '05', 'final']:
        validation['spiking_validation_cpu_' + n] = 'legacy__controlled_carlsim_cpu__r' + ('06' if n == 'final' else n)
    for old, new in validation.items():
        add('outputs/' + old, 'experiments/validation/' + new, 'validation')

    for old, new in [('grating_one_dat_preview_check_01', 'legacy__grating_single_preview__r01'),
                     ('grating_sequence_preview_check_01', 'legacy__grating_sequence_preview__r01'),
                     ('mt_video_fixture_01', 'legacy__mt_video_fixture__r01')]:
        add('outputs/' + old, 'experiments/scratch/' + new, 'scratch')

    # These "video" folders contain the only local stage arrays: keep those
    # arrays with their hashed metadata, and put rendering derivatives elsewhere.
    for old, rid in [('grating_mt_video_01', 'legacy__grating_80x80_f004800__mt_stages__r01'),
                     ('mt_stage_preview01', 'legacy__grating_80x80_f000300__mt_stages__r01')]:
        for p in sorted((ROOT / 'outputs' / old).iterdir()):
            response = p.name in ('stages.f32', 'stages.json')
            base = f'outputs/recordings/grating/{rid}' if response else f'outputs/visualizations/grating/{rid}/mt_stages__v01'
            add(p.relative_to(ROOT).as_posix(), base + '/' + p.name, 'recording' if response else 'visualization')

    selfmotion = bundles['outputs/selfmotion_01']
    add('outputs/selfmotion_mp4_01', 'input/stimuli/' + selfmotion + '/preview', 'stimulus_preview')
    add('outputs/calibration_texture_01_preview.png', 'input/stimuli/' + bundles['outputs/calibration_texture_01'] + '/preview/contact_sheet.png', 'stimulus_preview')
    add('outputs/validation_sequences_01/README.md', 'input/stimuli/sequence_preparation/legacy_v01/README.md', 'stimulus_metadata')
    add('outputs/validation_sequences_01/verification.json', 'input/stimuli/sequence_preparation/legacy_v01/verification.json', 'stimulus_metadata')

    for source, seq in [('walking', 'daria_walk'), ('paragliding', 'davis_paragliding'), ('snowboard', 'snowboard')]:
        add(f'srcipts/carlsimMT/{source}/spk_MT1CDS.dat',
            f'outputs/recordings/{seq}/legacy__{seq}__carlsim_import__r01/spikes/spk_MT1CDS.dat', 'recording_import')
    for source, seq in [('davis_paraliging', 'davis_paragliding'), ('snowboard', 'snowboard')]:
        for p in sorted((ROOT / 'srcipts/carlsimMT' / source).iterdir()):
            if p.name == 'spk_MT1CDS.dat':
                continue
            add(p.relative_to(ROOT).as_posix(), f'input/sources/{seq}/legacy_frames_v01/' + p.name, 'source_frames')

    # Compiled CMake trees retain embedded old paths; they are preserved as build
    # snapshots. New builds must use a fresh directory, never cmake --build these.
    for name in ('build_carlsim_mt', 'carlsim_cpu_build'):
        add('outputs/' + name, 'build/legacy_20261107/' + name, 'build_snapshot')
    add('outputs/export_fixture.exe', 'build/legacy_20261107/export_fixture.exe', 'build_snapshot')
    add('outputs/matlab_preferences', 'cache/matlab/preferences_legacy_20261107', 'local_state')
    add('MathWorks', 'cache/matlab/MathWorks_legacy_20261107', 'local_state')
    for p in sorted((ROOT / 'outputs').iterdir()):
        if p.is_file() and ('.tar.gz' in p.name or '.zip' in p.name):
            add(p.relative_to(ROOT).as_posix(), 'exports/legacy_20261107/' + p.name, 'export')
    for name in ('grating_one_dat_run.log', 'grating_sequence_run.log', 'matlab_bw_grating_01_run.log', 'matlab_bw_grating_02_run.log'):
        add('outputs/' + name, 'experiments/scratch/legacy__generation_logs__r01/' + name, 'scratch')
    add('outputs/python_spiking_init80.json', 'experiments/validation/legacy__python_spiking_init80__r01/report.json', 'validation')
    add('outputs/mt_stages.mp4', 'outputs/visualizations/unresolved/legacy__mt_stages__v01/mt_stages.mp4', 'unresolved')
    for name in ('davis_paragliding_video_70_01', 'snowboard_video_65_01'):
        add('outputs/' + name, 'experiments/scratch/legacy__empty_' + name, 'empty_attempt')
    return jobs


def enumerate_files(jobs):
    records = []
    seen = set()
    targets = set()
    for job in jobs:
        source = checked(job['old'])
        files = sorted(source.rglob('*')) if source.is_dir() else [source]
        for p in files:
            if p.is_symlink():
                raise ValueError(f'symlink in migration source: {p}')
            if not p.is_file():
                continue
            old = p.relative_to(ROOT).as_posix()
            suffix = p.relative_to(source) if source.is_dir() else Path()
            new = (Path(job['new']) / suffix).as_posix()
            target = checked(new)
            if old in seen or new in targets or target.exists():
                raise ValueError(f'duplicate source/target: {old} -> {new}')
            seen.add(old)
            targets.add(new)
            records.append(dict(old_path=old, new_path=new, bytes=p.stat().st_size,
                                sha256=sha(p), kind=job['kind']))
    return records


def index_stimuli():
    records = []
    # Validation fixtures stay inside their immutable suite; they are indexed
    # but not promoted to the reusable input library.
    for base in ('input/stimuli', 'experiments/validation', 'experiments/scratch'):
        for p in sorted((ROOT / base).rglob('*.dat')):
            with p.open('rb') as f:
                h = f.read(25)
            if len(h) != 25 or struct.unpack('<i', h[:4])[0] != 293390619:
                continue
            _, _, _, channels, w, height, frames = struct.unpack('<ifiBiii', h)
            if p.stat().st_size != 25 + channels * w * height * frames:
                raise ValueError(f'invalid stimulus size: {p}')
            records.append(dict(path=p.relative_to(ROOT).as_posix(), width=w,
                                height=height, frames=frames, channels=channels,
                                sha256=sha(p), scope='reusable' if base.startswith('input') else 'fixture'))
    write_csv(ROOT / 'input/registry.csv', records,
              ['path', 'width', 'height', 'frames', 'channels', 'sha256', 'scope'])
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    jobs = plan()
    if not args.apply:
        print(json.dumps(jobs, ensure_ascii=False, indent=2))
        return
    if AUDIT.exists():
        raise FileExistsError('migration already started; inspect journal, do not rerun blindly')
    records = enumerate_files(jobs)
    write_json(AUDIT / 'plan.json', jobs)
    write_csv(AUDIT / 'paths.csv', records, ['old_path', 'new_path', 'bytes', 'sha256', 'kind'])
    with (AUDIT / 'journal.jsonl').open('x', encoding='utf-8') as journal:
        for job in jobs:
            source, target = checked(job['old']), checked(job['new'])
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError(target)
            source.rename(target)
            journal.write(json.dumps(job, ensure_ascii=False) + '\n')
            journal.flush()
    for i, row in enumerate(records):
        p = checked(row['new_path'])
        if p.stat().st_size != row['bytes'] or sha(p) != row['sha256']:
            raise ValueError(f'migration integrity failure: {p}')
        if i % 200 == 0:
            print(f'[verified] {i}/{len(records)}', flush=True)
    stimuli = index_stimuli()
    rows = []
    for job in jobs:
        if job['kind'] in ('recording', 'recording_import', 'validation', 'visualization', 'comparison', 'unresolved'):
            path = checked(job['new'])
            if path.is_dir():
                rows.append(dict(id=path.name, kind=job['kind'], path=job['new'],
                                 legacy_path=job['old'], status='legacy_imported',
                                 parent_run_ids='', stimulus_ids='', notes='原始证据保留；关联见artifact.json；导入不等于通过验收'))
    write_csv(ROOT / 'experiments/registry.csv', rows,
              ['id', 'kind', 'path', 'legacy_path', 'status', 'parent_run_ids', 'stimulus_ids', 'notes'])
    write_json(AUDIT / 'verification.json', dict(complete=True, moved_operations=len(jobs),
               verified_files=len(records), verified_bytes=sum(r['bytes'] for r in records),
               stimulus_files=len(stimuli), content_changed=False,
               note='All moved files including historical JSON/CSV/binaries retain their SHA256. New indexes are additive.'))
    # Remove only now-empty legacy containers, never recursive deletion.
    for name in ('outputs/stimuli', 'outputs/validation_sequences_01', 'srcipts/bw_grating_output',
                 'srcipts/carlsimMT/walking', 'srcipts/carlsimMT/paragliding',
                 'srcipts/carlsimMT/snowboard', 'srcipts/carlsimMT/davis_paraliging',
                 'srcipts/carlsimMT', 'outputs/grating_mt_video_01', 'outputs/mt_stage_preview01'):
        p = checked(name)
        if p.is_dir() and not any(p.iterdir()):
            p.rmdir()
    print(json.dumps(json.loads((AUDIT / 'verification.json').read_text(encoding='utf-8')), indent=2))


if __name__ == '__main__':
    main()
