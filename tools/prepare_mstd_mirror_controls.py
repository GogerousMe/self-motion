"""Prepare a strictly pixel-mirrored +/-7deg diagnostic pair and MP4 previews."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from mstd.calibration import inspect_stimulus, write_stimulus
from mstd.mirror_controls import reflect_image, reflect_truth
from mstd.selfmotion import Camera, Motion, sequence

BUNDLE = 'mstd_mirror_s0211__80x80__f000080__dt50ms__2cases__v01'
FREEZE = 'experiments/validation/20261109__mstd_geometry_gate_80x80_f000080__py_rate_cpu__r01/frozen_validity.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def preview(ffmpeg, path, pixels, width):
    subprocess.run([ffmpeg, '-hide_banner','-loglevel','error','-n','-f','rawvideo',
                    '-pixel_format','gray','-video_size',f'{width}x80','-framerate','20','-i','pipe:0',
                    '-vf',f'scale={width*8}:640:flags=neighbor','-an','-c:v','libx264',
                    '-pix_fmt','yuv420p','-crf','18','-movflags','+faststart',str(path)],
                   input=pixels, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='input/stimuli/selfmotion/'+BUNDLE)
    p.add_argument('--frozen-config', default=FREEZE)
    a = p.parse_args()
    folder = (ROOT/a.output).resolve()
    if not folder.is_relative_to(ROOT/'input/stimuli/selfmotion'):
        p.error('output must be under input/stimuli/selfmotion')
    freeze_path = ROOT/a.frozen_config
    frozen = json.loads(freeze_path.read_text(encoding='utf-8'))
    if not frozen['development_passed'] or frozen['validity']['mode']!='geometry_support':
        p.error('use qualified frozen geometry_support parameters')
    ffmpeg = shutil.which('ffmpeg')
    ffprobe = shutil.which('ffprobe')
    if not ffmpeg or not ffprobe:
        p.error('ffmpeg and ffprobe are required for verified MP4 previews')
    with (ROOT/'input/registry.csv').open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f); input_fields=reader.fieldnames; indexed=list(reader)
    with (ROOT/'experiments/registry.csv').open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f); fields=reader.fieldnames; registered=list(reader)
    if any(r['id']==folder.name for r in registered):
        p.error('bundle ID already registered')
    folder.mkdir(parents=True,exist_ok=False); (folder/'preview').mkdir()
    camera=Camera(); right_motion=Motion(heading_deg=7)
    right=list(sequence(camera,right_motion,'cloud',frames=80,frame_ms=50,points=250,seed=211))
    left=[(reflect_image(image,80,80),reflect_truth(truth,80)) for image,truth in right]
    definitions=[('cloud_heading_left7',left,Motion(heading_deg=-7)),
                 ('cloud_heading_right7',right,right_motion)]
    runs=[]; input_rows=[]
    for name,rows,motion in definitions:
        path=folder/(name+'.dat')
        write_stimulus(path,80,80,(image for image,truth in rows))
        with (folder/(name+'.truth.jsonl')).open('x',encoding='utf-8') as f:
            for image,truth in rows:
                f.write(json.dumps(truth,allow_nan=False)+'\n')
        shape=inspect_stimulus(path)
        assert (shape['width'],shape['height'],shape['frames'],shape['channels'])==(80,80,80,1)
        assert all(abs(truth['camera']['heading_camera_deg']-motion.heading_deg)<1e-9 for image,truth in rows)
        preview(ffmpeg,folder/'preview'/(name+'.mp4'),b''.join(image for image,truth in rows),80)
        rel=path.relative_to(ROOT).as_posix()
        if any(r['path']==rel for r in indexed):raise ValueError('already indexed path')
        input_rows.append(dict(path=rel,width=80,height=80,frames=80,channels=1,sha256=sha(path),scope='reusable'))
        runs.append(dict(name=name,scene='cloud',motion=asdict(motion),stimulus=path.name,
                         stimulus_sha256=sha(path),truth=name+'.truth.jsonl',video='preview/'+name+'.mp4',
                         current_backend='translation_baseline'))
    r=np.frombuffer(b''.join(image for image,truth in right),np.uint8).reshape(80,80,80)
    l=np.frombuffer(b''.join(image for image,truth in left),np.uint8).reshape(80,80,80)
    assert np.array_equal(l,r[:,:,::-1])
    for (_,lt),(_,rt) in zip(left,right):
        assert lt['time_s']==rt['time_s'] and lt['camera']['vx_mps']==-rt['camera']['vx_mps']
        for lp,rp in zip(lt['sparse_flow_to_next_frame'],rt['sparse_flow_to_next_frame']):
            assert lp['id']==rp['id'] and abs(lp['u_px']+rp['u_px']-79)<1e-12
            assert lp['v_px']==rp['v_px'] and lp['depth_m']==rp['depth_m']
            assert lp['du_px']==(-rp['du_px'] if rp['du_px'] is not None else None)
    pair=folder/'preview/left_minus7__right_plus7.mp4'
    preview(ffmpeg,pair,np.concatenate((l,r),axis=2).tobytes(),160)
    video_checks={}
    for path in (folder/'preview').glob('*.mp4'):
        stream=json.loads(subprocess.check_output([ffprobe,'-v','error','-select_streams','v:0',
            '-show_entries','stream=codec_name,width,height,nb_frames,avg_frame_rate,duration','-of','json',str(path)],text=True))['streams'][0]
        assert stream['codec_name']=='h264' and stream['nb_frames']=='80' and stream['avg_frame_rate']=='20/1' and float(stream['duration'])==4
        video_checks[path.name]=stream
    dump(folder/'intrinsics.json',camera.intrinsics)
    manifest=dict(schema='selfmotion-stimulus-v1',width=80,height=80,frames=80,frame_ms=50,duration_ms=4000,
        seed=211,points=250,camera=asdict(camera),intrinsics_px=camera.intrinsics,runs=runs,
        protocol_scope='diagnostic_development',validity_freeze_sha256=sha(freeze_path),
        mirror_pair=dict(left='cloud_heading_left7',right='cloud_heading_right7',axis='image x',center_px=39.5,
                         pixel_relation='left[t,y,x] == right[t,y,79-x]',truth_relation='u_left=79-u_right; du_left=-du_right; heading_left=-heading_right'),
        suggested_neural_seeds=[43,44,45],
        source_hashes={s:sha(ROOT/s) for s in ('tools/prepare_mstd_mirror_controls.py','src/mstd/mirror_controls.py','src/mstd/selfmotion.py')},
        notes=['Left image/truth are reflected from the SAME right sequence; not separately rendered worlds.',
               'Independent scene seed211; developmental control after inspecting scene97, not a final blind benchmark.',
               'Each case resets the model. Same neural seed does not mirror realized neuron spike samples.',
               'DAT is plain grayscale imagery. Truth is external to the model. Preview labels/order are not input pixels.'])
    dump(folder/'manifest.json',manifest)
    dump(folder/'verification.json',dict(pixel_mirror_exact=True,truth_reflection_checked=True,
        input_shapes='80x80 grayscale, 80 frames, 25-byte header',previews=video_checks))
    artifact=dict(schema='mstd-artifact-v1',id=folder.name,kind='stimulus_bundle',status='completed',
        stimulus_ids=[folder.name+'__'+x['name'] for x in runs],input_paths=[x['path'] for x in input_rows],
        input_sha256={x['path']:x['sha256'] for x in input_rows},parent_run_ids=[],
        validity_freeze_sha256=sha(freeze_path),notes='Exact pixel mirror +/-7deg, scene211; developmental multi-neural-seed control, responses pending.')
    dump(folder/'artifact.json',artifact)
    with (ROOT/'input/registry.csv').open('a',encoding='utf-8',newline='') as f:csv.DictWriter(f,fieldnames=input_fields).writerows(input_rows)
    with (ROOT/'experiments/registry.csv').open('a',encoding='utf-8',newline='') as f:
        csv.DictWriter(f,fieldnames=fields).writerow(dict(id=folder.name,kind='stimulus_bundle',path=folder.relative_to(ROOT).as_posix(),
            status='completed',stimulus_ids=';'.join(artifact['stimulus_ids']),notes=artifact['notes']))
    print(json.dumps(dict(output=str(folder),cases=2,pixel_mirror_exact=True,neural_runs_planned=6)))


if __name__=='__main__':
    main()
