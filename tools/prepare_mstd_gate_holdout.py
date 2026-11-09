"""Create independently seeded image DATs after qualifying/fixing the gate.

Retains the project's 25-byte VisualStimulus header, raw gray pixels and JSONL
ground truth. MP4s are previews only; the network consumes unannotated DATs.
"""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from mstd.calibration import inspect_stimulus, write_stimulus
from mstd.selfmotion import Camera, Motion, sequence


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='input/stimuli/selfmotion/mstd_gate_holdout_s0097__80x80__f000080__dt50ms__7cases__v01')
    p.add_argument('--frozen-config', required=True)
    a = p.parse_args()
    frozen_path = ROOT/a.frozen_config
    frozen = json.loads(frozen_path.read_text(encoding='utf-8'))
    if not frozen['development_passed']:
        p.error('development did not qualify this gate')
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        p.error('ffmpeg is required to save MP4 previews')
    folder = (ROOT/a.output).resolve()
    if not folder.is_relative_to(ROOT/'input/stimuli/selfmotion'):
        p.error('output must be inside input/stimuli/selfmotion')
    with (ROOT/'input/registry.csv').open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f); input_fields = reader.fieldnames; indexed = list(reader)
    with (ROOT/'experiments/registry.csv').open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f); experiment_fields = reader.fieldnames; registered = list(reader)
    if any(r['id']==folder.name for r in registered):
        p.error('bundle already registered')
    folder.mkdir(parents=True, exist_ok=False)
    (folder/'preview').mkdir()
    camera = Camera(); seed = 97; runs = []; input_rows = []
    definitions = [('cloud_static','cloud',Motion('static')),
                   ('cloud_forward','cloud',Motion()),
                   ('cloud_heading_left7','cloud',Motion(heading_deg=-7)),
                   ('cloud_heading_right7','cloud',Motion(heading_deg=7)),
                   ('ground_forward','ground',Motion()),
                   ('ground_heading_left10','ground',Motion(heading_deg=-10)),
                   ('ground_heading_right10','ground',Motion(heading_deg=10))]
    for name, scene, motion in definitions:
        path = folder/(name+'.dat')
        rows = list(sequence(camera, motion, scene, frames=80, frame_ms=50, points=250, seed=seed))
        write_stimulus(path, 80, 80, (image for image, truth in rows))
        with (folder/(name+'.truth.jsonl')).open('x', encoding='utf-8') as f:
            for image, truth in rows:
                f.write(json.dumps(truth, allow_nan=False)+'\n')
        shape = inspect_stimulus(path)
        assert (shape['width'],shape['height'],shape['frames'],shape['channels']) == (80,80,80,1)
        if motion.mode == 'static':
            assert all(image == rows[0][0] for image, truth in rows)
        elif motion.heading_deg != 0:
            assert all(abs(truth['camera']['heading_camera_deg']-motion.heading_deg) < 1e-9 for image, truth in rows)
        preview = folder/'preview'/(name+'.mp4')
        subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-n','-f','rawvideo',
                        '-pixel_format','gray','-video_size','80x80','-framerate','20','-i','pipe:0',
                        '-vf','scale=640:640:flags=neighbor','-an','-c:v','libx264','-pix_fmt','yuv420p',
                        '-crf','18','-movflags','+faststart',str(preview)],
                       input=b''.join(image for image, truth in rows), check=True)
        source_hash = sha(path)
        relative = path.relative_to(ROOT).as_posix()
        if any(r['path']==relative for r in indexed):
            raise ValueError('stimulus path already indexed')
        input_rows.append(dict(path=relative, width=80, height=80, frames=80, channels=1, sha256=source_hash, scope='reusable'))
        runs.append(dict(name=name, scene=scene, motion=asdict(motion), stimulus=path.name,
                         stimulus_sha256=source_hash, truth=name+'.truth.jsonl',
                         video='preview/'+name+'.mp4', current_backend='translation_baseline' if motion.mode=='straight' else 'static_control'))
        print(f'Prepared {name}: 80 frames, {source_hash[:12]}', flush=True)
    dump(folder/'intrinsics.json', camera.intrinsics)
    dump(folder/'manifest.json', dict(schema='selfmotion-stimulus-v1', width=80, height=80, frames=80,
         frame_ms=50, duration_ms=4000, seed=seed, points=250, camera=asdict(camera),
         intrinsics_px=camera.intrinsics, runs=runs,
         coordinates='world x-right y-up z-initial-forward; image x-right y-down',
         validity_freeze_sha256=sha(frozen_path),
         source_hashes={f:sha(ROOT/f) for f in ('src/mstd/selfmotion.py','tools/prepare_mstd_gate_holdout.py')},
         notes=['Independent scene seed97; old suite used seed0. No neural outputs used to construct these inputs.',
                'Every case resets the model. Same scene seed across conditions controls scene content.',
                'DAT contains only gray images; point depths/flow are external ground truth, never model inputs.',
                '20Hz, 80 frames per independent condition; previews H264 MP4.']))
    artifact = dict(schema='mstd-artifact-v1', id=folder.name, kind='stimulus_bundle', status='completed',
                    stimulus_ids=[folder.name+'__'+r['name'] for r in runs],
                    input_paths=[r['path'] for r in input_rows],
                    input_sha256={r['path']:r['sha256'] for r in input_rows}, parent_run_ids=[],
                    validity_freeze_sha256=sha(frozen_path), notes='Independent scene seed97, post-freeze full image-chain holdout; responses pending.')
    dump(folder/'artifact.json', artifact)
    with (ROOT/'input/registry.csv').open('a', encoding='utf-8', newline='') as f:
        csv.DictWriter(f, fieldnames=input_fields).writerows(input_rows)
    with (ROOT/'experiments/registry.csv').open('a', encoding='utf-8', newline='') as f:
        csv.DictWriter(f, fieldnames=experiment_fields).writerow(dict(id=folder.name, kind='stimulus_bundle',
          path=folder.relative_to(ROOT).as_posix(), status='completed', stimulus_ids=';'.join(artifact['stimulus_ids']), notes=artifact['notes']))
    print(json.dumps(dict(output=str(folder), runs=len(runs), scene_seed=seed)))


if __name__ == '__main__':
    main()
