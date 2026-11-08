"""Verify relocated artifacts and enrich additive associations, not old reports."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'docs/migrations/20261107_layout'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh-associations', action='store_true')
    parser.add_argument('--hash-all', action='store_true')
    args = parser.parse_args()
    mapping = rows(AUDIT / 'paths.csv')
    stimuli = rows(ROOT / 'input/registry.csv')
    indexed = {}
    for r in stimuli:
        indexed.setdefault(r['sha256'], []).append(r)
    registry = rows(ROOT / 'experiments/registry.csv')
    artifacts = [(r, ROOT / r['path'] / 'artifact.json') for r in registry]
    for r, path in artifacts:
        a = read(path)
        if args.refresh_associations:
            hashes = set(a.get('input_sha256', []))
            parents = set(a.get('parent_paths', []))
            for parent in parents:
                report = ROOT / parent / 'model_report.json'
                if report.is_file():
                    h = read(report).get('input', {}).get('stimulus_sha256')
                    if h:
                        hashes.add(h)
            # The suite is the stable run ID; d000 is a condition within it.
            a['parent_run_ids'] = sorted({Path(p).parts[2] if p.startswith('experiments/validation/') else Path(p).name for p in parents})
            entries = []
            for h in sorted(hashes):
                candidates = indexed.get(h, [])
                local = [x for x in candidates if any(x['path'].startswith(p + '/') for p in parents)]
                if candidates:
                    entries.append((local or candidates)[0])
            a['input_sha256'] = sorted(hashes)
            a['input_paths'] = [e['path'] for e in entries]
            a['stimulus_ids'] = [Path(e['path']).stem if Path(e['path']).stem == Path(e['path']).parent.name else Path(e['path']).parent.name + '__' + Path(e['path']).stem for e in entries]
            if a['kind'] == 'comparison':
                report = read(path.parent / 'comparison_report.json')
                expected = report.get('spikes_sha256')
                linked_spikes = [ROOT / p / 'spikes/spk_MT1CDS.dat' for p in parents]
                if expected and not any(p.is_file() and digest(p) == expected for p in linked_spikes):
                    a['unresolved_inputs'] = [dict(kind='carlsim_spikes', sha256=expected,
                         status='exact_historical_payload_not_locally_paired')]
                    a['notes'] += ' 比较报告的CARLsim脉冲哈希未与当前本机导入文件匹配；原对比证据保留，不将现存脉冲误认作原父输入。'
            path.write_text(json.dumps(a, indent=2, ensure_ascii=False), encoding='utf-8')
            r['parent_run_ids'] = ';'.join(a['parent_run_ids'])
            r['stimulus_ids'] = ';'.join(a['stimulus_ids'])
            r['notes'] = a['notes']
        for key in ('input_paths', 'parent_paths'):
            for relative in a.get(key, []):
                target = (ROOT / relative).resolve()
                if not target.is_relative_to(ROOT) or not target.exists():
                    raise ValueError(f'broken/escaping artifact link: {path}: {relative}')
    if args.refresh_associations:
        with (ROOT / 'experiments/registry.csv').open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(registry[0]))
            writer.writeheader()
            writer.writerows(registry)
    ids = [r['id'] for r in registry]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate artifact IDs')
    for row in mapping:
        p = ROOT / row['new_path']
        if not p.is_file() or p.stat().st_size != int(row['bytes']):
            raise ValueError(f'moved file missing/changed size: {p}')
        if args.hash_all and digest(p) != row['sha256']:
            raise ValueError(f'moved file hash mismatch: {p}')
    for row in stimuli:
        if digest(ROOT / row['path']) != row['sha256']:
            raise ValueError(f'stimulus hash mismatch: {row["path"]}')
    result = dict(complete=True, moved_files_checked=len(mapping), hash_all=args.hash_all,
                  stimulus_hashes_checked=len(stimuli), artifact_links_checked=len(artifacts),
                  unresolved_artifacts=[r['path'] for r, p in artifacts if read(p).get('unresolved_inputs') or read(p)['status'] == 'unresolved'])
    (AUDIT / 'final_audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
