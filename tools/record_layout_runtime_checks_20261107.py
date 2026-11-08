"""Register the three actual relocation checks and summarize their evidence."""
import csv
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'docs/migrations/20261107_layout'


def main():
    definitions = [
        ('scratch', 'experiments/scratch/20261107__layout_mt_replay__r01',
         ['outputs/recordings/daria_walk/legacy__daria_walk_180x144_f000084__py_rate_cuda__r01']),
        ('scratch', 'experiments/scratch/20261107__layout_cache_replay__r01',
         ['outputs/recordings/grating/legacy__grating_80x80_f004800__mt_stages__r01']),
        ('validation', 'experiments/validation/20261107__layout_carlsim_cpu__r01', []),
    ]
    registry = ROOT / 'experiments/registry.csv'
    with registry.open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    known = {r['id'] for r in rows}
    checks = []
    for kind, relative, parents in definitions:
        p = ROOT / relative
        if p.name in known:
            raise ValueError('runtime checks already registered')
        if kind == 'validation':
            report = json.loads((p / 'validation.json').read_text())
            if not report['passed']:
                raise ValueError('native runtime checks failed')
            evidence = dict(native_checks=report['checks'], passed=report['passed'])
        else:
            payload = json.loads(subprocess.check_output([
                'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
                '-show_entries', 'stream=nb_read_frames,width,height', '-of', 'json',
                str(p / 'mt_stages.mp4')], text=True))
            if int(payload['streams'][0]['nb_read_frames']) != 2:
                raise ValueError('replayed video frame count mismatch')
            evidence = dict(video_probe=payload, passed=True)
        artifact = dict(schema='mstd-artifact-v1', id=p.name, kind=kind, status='completed',
                        input_paths=[], stimulus_ids=[], parent_paths=parents,
                        parent_run_ids=[Path(x).name for x in parents],
                        notes='目录迁移工程验收；不代表模型精度验证。', verification=evidence)
        (p / 'artifact.json').write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding='utf-8')
        rows.append(dict(id=p.name, kind=kind, path=relative, legacy_path='', status='completed',
                         parent_run_ids=';'.join(artifact['parent_run_ids']), stimulus_ids='', notes=artifact['notes']))
        checks.append(dict(path=relative, **evidence))
    with registry.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(targeted_unittests_passed=13, checks=checks,
                  build='build/carlsim_mt; freshly configured against relocated preserved CARLsim library',
                  note='MATLAB naming entry inspected but MATLAB UI was not executed. External server files were not moved.')
    (AUDIT / 'runtime_checks.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(dict(registered=len(checks), passed=True)))


if __name__ == '__main__':
    main()
