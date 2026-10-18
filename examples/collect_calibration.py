"""Run an existing C++ binary that honors V1_CALIBRATION_INPUT/OUTPUT.

Pure stdlib. Never edits C++ sources or overwrites a response directory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', required=True)
    parser.add_argument('--executable', required=True)
    parser.add_argument('--workdir', required=True)
    parser.add_argument('--library', help='optional actual MotionEnergy build library to fingerprint')
    parser.add_argument('--runs', nargs='+', help='optional named subset; default all runs')
    args = parser.parse_args()
    suite = Path(args.suite).resolve()
    executable, workdir = Path(args.executable).resolve(), Path(args.workdir).resolve()
    manifest = suite/'manifest.json'
    meta = json.loads(manifest.read_text(encoding='utf-8'))
    if meta.get('schema') not in ('v1-direction-calibration-v1','selfmotion-stimulus-v1') or meta['frame_ms'] != 50:
        parser.error('this collection recipe requires a v1 manifest with 50ms frames')
    if not executable.is_file() or not workdir.is_dir():
        parser.error('executable and working directory must exist')
    names = {run['name'] for run in meta['runs']}
    if args.runs and (not set(args.runs) <= names or len(set(args.runs)) != len(args.runs)):
        parser.error('unknown or duplicate run names')
    selected = [run for run in meta['runs'] if not args.runs or run['name'] in args.runs]
    for run in selected:
        if run.get('spikes'):
            parser.error('suite already contains responses; use a fresh suite for a new acquisition')
        if Path(run['name']).name != run['name'] or not run['name']:
            parser.error('invalid run name')
        stimulus = (suite/run['stimulus']).resolve()
        if stimulus.parent != suite or sha256(stimulus) != run['stimulus_sha256']:
            parser.error('stimulus location/checksum mismatch')
    acquisition = {'executable_sha256':sha256(executable),
                   'library_sha256':sha256(args.library) if args.library else None,
                   'timing_requirement':'operator verified 50ms per frame',
                   'library_note':'hash records supplied file, not proof of dynamic loader selection'}
    for run in selected:
        output = suite/'responses'/run['name']
        output.mkdir(parents=True,exist_ok=False)
        spikes = output/'spk_V1.dat'
        env = os.environ.copy()
        env['V1_CALIBRATION_INPUT'] = str((suite/run['stimulus']).resolve())
        env['V1_CALIBRATION_OUTPUT'] = str(spikes)
        with (output/'process.log').open('w',encoding='utf-8') as log:
            subprocess.run([str(executable)],cwd=workdir,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        if not spikes.is_file():
            raise RuntimeError('No requested spike file: check C++ environment-variable integration; see process.log')
        with spikes.open('rb') as stream:
            raw = stream.read(20)
            if len(raw) != 20:
                raise ValueError('truncated response header')
            signature,version,w,h,depth = struct.unpack('<ifiii',raw)
        if signature != 206661989 or abs(version-.2)>1e-6 or (w,h,depth)!=(meta['width'],meta['height'],84):
            raise ValueError('unexpected response format/grid; verify modified V1 library')
        if (spikes.stat().st_size-20)%8:
            raise ValueError('truncated response event')
        run['spikes'] = spikes.relative_to(suite).as_posix()
        run['acquisition'] = {**acquisition,'spikes_sha256':sha256(spikes)}
        # Save every successful run. On failure preserve logs and collected data.
        manifest.write_text(json.dumps(meta,indent=2),encoding='utf-8')
        print('Collected:',run['name'],flush=True)
    print('Return this complete suite directory, including responses/ and manifest.json.')


if __name__ == '__main__':
    main()
