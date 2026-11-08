"""Add relocation associations and update current documentation after migration.

Does not rewrite historical data or reports. One-time tool, refuses to overwrite
artifact metadata. All text replacements are restricted to current source/docs.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'docs/migrations/20261107_layout'


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)


def csv_rows(path):
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def main():
    if not load(AUDIT / 'verification.json')['complete']:
        raise ValueError('complete data verification first')
    paths = csv_rows(AUDIT / 'paths.csv')
    stimuli = csv_rows(ROOT / 'input/registry.csv')
    by_hash = {}
    for row in stimuli:
        by_hash.setdefault(row['sha256'], row)
    jobs = load(AUDIT / 'plan.json')
    replacements = {j['old']: j['new'] for j in jobs}

    def relocate(text):
        text = text.replace('\\', '/')
        prefix = ROOT.as_posix() + '/'
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):]
        for row in paths:
            if text == row['old_path']:
                return row['new_path']
        for old, new in sorted(replacements.items(), key=lambda p: -len(p[0])):
            if text == old or text.startswith(old + '/'):
                return new + text[len(old):]
        return None

    def stimulus_id(row):
        p = Path(row['path'])
        if p.stem == p.parent.name:
            return p.stem
        return p.parent.name + '__' + p.stem

    # Five standalone packages and each reusable bundle get additive metadata.
    groups = {}
    for row in stimuli:
        if row['scope'] != 'reusable':
            continue
        p = Path(row['path'])
        parts = p.parts
        depth = 5 if parts[2] == 'natural' else 4
        group = Path(*parts[:depth])
        groups.setdefault(group, []).append(row)
    for group, entries in groups.items():
        write(ROOT / group / 'artifact.json', dict(
            schema='mstd-artifact-v1', id=group.name, kind='stimulus_bundle' if len(entries) > 1 else 'stimulus',
            status='legacy_imported', stimulus_ids=[stimulus_id(r) for r in entries],
            input_paths=[r['path'] for r in entries],
            stimuli=[dict(path=r['path'], sha256=r['sha256'], width=int(r['width']), height=int(r['height']), frames=int(r['frames'])) for r in entries],
            notes='DAT头与字节数已校验；帧间隔以配套元数据为准。套件内部协议文件名保持不变。'))

    products = []
    for category, pattern in [('recording', 'outputs/recordings/*/*'),
                              ('visualization', 'outputs/visualizations/*/*/*'),
                              ('comparison', 'outputs/comparisons/*/*'),
                              ('validation', 'experiments/validation/*'),
                              ('scratch', 'experiments/scratch/*')]:
        products.extend((category, p) for p in ROOT.glob(pattern) if p.is_dir())
    unresolved = ROOT / 'outputs/visualizations/unresolved/legacy__mt_stages__v01'
    products.append(('unresolved', unresolved))
    rows = []
    for category, p in sorted(products, key=lambda x: str(x[1])):
        # Frame directories in unresolved aren't logical products.
        if p == unresolved / 'mt_stages.mp4':
            continue
        relative = p.relative_to(ROOT).as_posix()
        aid = (p.parent.name + '/' + p.name) if category == 'visualization' else p.name
        hashes = set()
        parent_paths = set()
        reports = []
        report_names = {'model_report.json', 'video_config.json', 'visualization.json', 'comparison_report.json', 'stages.json'}
        for f in p.rglob('*.json'):
            if f.name not in report_names:
                continue
            report = load(f)
            reports.append(report)
            for h in (report.get('input_sha256'), report.get('stimulus_sha256'), report.get('input', {}).get('stimulus_sha256')):
                if h:
                    hashes.add(h)
            old_recording = report.get('recording')
            if old_recording:
                moved = relocate(old_recording)
                if moved:
                    parent_paths.add(moved)
        if category == 'visualization':
            rid, dataset = p.parent.name, p.parent.parent.name
            candidates = [ROOT / 'outputs/recordings' / dataset / rid,
                          ROOT / 'experiments/validation' / rid]
            for candidate in candidates:
                if candidate.is_dir():
                    parent_paths.add(candidate.relative_to(ROOT).as_posix())
        if category == 'comparison':
            # Pair Python runs by exact input SHA256; CARLsim import by sequence.
            dataset = p.parent.name
            for candidate in (ROOT / 'outputs/recordings' / dataset).iterdir():
                if (candidate / 'model_report.json').is_file():
                    h = load(candidate / 'model_report.json')['input'].get('stimulus_sha256')
                    if h in hashes:
                        parent_paths.add(candidate.relative_to(ROOT).as_posix())
                elif (candidate / 'spikes/spk_MT1CDS.dat').is_file():
                    from hashlib import sha256
                    spike_hash = sha256((candidate / 'spikes/spk_MT1CDS.dat').read_bytes()).hexdigest()
                    if any(r.get('spikes_sha256') == spike_hash for r in reports):
                        parent_paths.add(candidate.relative_to(ROOT).as_posix())
        parent_ids = sorted(Path(path).name for path in parent_paths)
        entries = [by_hash[h] for h in sorted(hashes) if h in by_hash]
        old_paths = sorted({r['old_path'] for r in paths if r['new_path'] == relative or r['new_path'].startswith(relative + '/')})
        status = 'unresolved' if category == 'unresolved' else 'legacy_imported'
        notes = '原始报告保留字节与旧路径；导入不代表通过验收。'
        if category == 'unresolved':
            notes += '此散落视频尚未可靠确认输入和父响应，禁止用于定量结论。'
        if category == 'visualization' and p.parent.name.startswith('legacy__grating_80x80'):
            notes += '原video_config.source_cache为历史相对路径；当前stages.json见parent_paths。'
        artifact = dict(schema='mstd-artifact-v1', id=aid, kind=category, status=status,
                        stimulus_ids=[stimulus_id(r) for r in entries], input_paths=[r['path'] for r in entries],
                        input_sha256=sorted(hashes), parent_run_ids=parent_ids, parent_paths=sorted(parent_paths),
                        legacy_paths=old_paths, notes=notes)
        write(p / 'artifact.json', artifact)
        rows.append(dict(id=aid, kind=category, path=relative,
                         legacy_path=';'.join(old_paths[:2]), status=status,
                         parent_run_ids=';'.join(parent_ids), stimulus_ids=';'.join(artifact['stimulus_ids']), notes=notes))
    registry = ROOT / 'experiments/registry.csv'
    # This index is new in this migration, not historical evidence.
    with registry.open('w', encoding='utf-8', newline='') as f:
        fields = ['id', 'kind', 'path', 'legacy_path', 'status', 'parent_run_ids', 'stimulus_ids', 'notes']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # Update current docs/scripts with exact move mappings. Historical validation
    # reports retain their recorded paths and link to the migration index instead.
    historical = {'VALIDATION.md', 'CALIBRATION_VALIDATION.md', 'SELFMOTION_VALIDATION.md',
                  'PYTHON_SPIKING_VALIDATION.md', 'SPIKING_MT_VALIDATION.md', 'M1_VALIDATION.md',
                  'PYTHON_V1_VALIDATION.md', 'MT_STAGE_VALIDATION.md'}
    current = [ROOT / 'README.md', ROOT / 'examples/prepare_validation_sequences.py']
    current += [p for p in (ROOT / 'docs').glob('*.md') if p.name not in historical and p.name != 'FILE_ORGANIZATION.md']
    current += [ROOT / 'matlab/README.md', ROOT / 'srcipts/README.md']
    changed = []
    for p in current:
        old = p.read_text(encoding='utf-8')
        text = old
        for before, after in sorted(replacements.items(), key=lambda x: -len(x[0])):
            text = text.replace(before, after)
        # Build paths in operational instructions refer to new builds, not snapshots.
        text = text.replace('outputs/build_carlsim_mt', 'build/carlsim_mt')
        if p.name == 'SPIKING_MT_PLAN.md':
            text = text.replace('build/legacy_20261107/build_carlsim_mt', 'build/carlsim_mt')
        if text != old:
            p.write_text(text, encoding='utf-8')
            changed.append(p.relative_to(ROOT).as_posix())
    for name in historical:
        p = ROOT / 'docs' / name
        if not p.is_file():
            continue
        text = p.read_text(encoding='utf-8')
        notice = '> 路径说明（2026-11-07）：本文保留历史实验路径；当前存放位置请查 [文件规范](FILE_ORGANIZATION.md) 和 [迁移映射](migrations/20261107_layout/paths.csv)。\n\n'
        p.write_text(notice + text, encoding='utf-8')
    write(AUDIT / 'associations.json', dict(artifact_count=len(products) + len(groups),
               experiment_rows=len(rows), stimulus_packages=len(groups), updated_current_docs=changed))
    print(json.dumps(dict(artifacts=len(products) + len(groups), registry_rows=len(rows), updated_docs=len(changed))))


if __name__ == '__main__':
    main()
