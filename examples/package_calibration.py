"""Package fresh stimulus suites and the collection guide for Linux transfer."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suites', nargs='+')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    files, kinds = [], set()
    for item in args.suites:
        suite = Path(item)
        meta = json.loads((suite/'manifest.json').read_text(encoding='utf-8'))
        kind = meta['kind']
        if kind not in ('grating','texture') or kind in kinds:
            raise ValueError('one fresh suite per stimulus kind required')
        kinds.add(kind)
        for run in meta['runs']:
            if run.get('spikes'):
                raise ValueError('only fresh suites without response data can be packaged')
            path = suite/run['stimulus']
            if path.resolve().parent != suite.resolve():
                raise ValueError('stimulus must reside directly in suite')
            if hashlib.sha256(path.read_bytes()).hexdigest() != run['stimulus_sha256']:
                raise ValueError('stimulus checksum mismatch')
            files.append((path,f'{kind}/{path.name}'))
        files.extend([(suite/'manifest.json',f'{kind}/manifest.json'),
                      (suite/'preview.pgm',f'{kind}/preview.pgm')])
    files.extend([(root/'examples/collect_calibration.py','collect_calibration.py'),
                  (root/'docs/CALIBRATION.md','COLLECTION.md')])
    output = Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as stream:
        with tarfile.open(fileobj=stream,mode='w:gz') as archive:
            for path,name in files:
                archive.add(path,arcname=name,recursive=False)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_name(output.name+'.sha256').write_text(digest+'  '+output.name+'\n',encoding='ascii')
    print(json.dumps({'archive':str(output),'sha256':digest,'files':len(files)}))


if __name__ == '__main__':
    main()
