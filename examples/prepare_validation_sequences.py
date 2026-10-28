"""Build two continuous input DATs: texture controls and five self-motion cases.

No raw DAT concatenation: one header is written and payloads are concatenated.
Original eight texture clips are preserved; seamless switches are generated separately.
"""
import argparse
import csv
import json
from math import cos, floor, pi, sin
from pathlib import Path
import random
import shutil

from mstd.calibration import file_sha256, inspect_stimulus, write_stimulus

BASELINE = ['cloud_static', 'cloud_forward', 'ground_forward',
            'ground_heading_left10', 'ground_heading_right10']


def read_images(path):
    info = inspect_stimulus(path)
    if info['channels'] != 1:
        raise ValueError('only single-channel grayscale source DATs supported')
    with Path(path).open('rb') as stream:
        stream.seek(25)
        for _ in range(info['frames']):
            yield stream.read(info['width']*info['height'])


def continuous_texture(w, h, frames_per_direction, speed, seed):
    rng = random.Random(seed)
    texture = [230 if rng.random() < .2 else 25 for _ in range(w*h)]
    sx = sy = 0.
    for direction in range(0, 360, 45):
        dx, dy = speed*cos(direction*pi/180), speed*sin(direction*pi/180)
        for index in range(frames_per_direction):
            # Position is carried across every direction boundary; no image reset.
            if direction or index:
                sx += dx
                sy += dy
            frame = bytearray(w*h)
            for y in range(h):
                for x in range(w):
                    u, v = x-sx, y-sy
                    ix, iy = floor(u), floor(v)
                    ax, ay = u-ix, v-iy
                    value = sum(texture[((iy+oy)%h)*w+(ix+ox)%w]*wx*wy
                                for ox,wx in ((0,1-ax),(1,ax)) for oy,wy in ((0,1-ay),(1,ay)))
                    frame[y*w+x] = max(0,min(255,floor(value+.5+1e-9)))
            yield direction, bytes(frame)


class Builder:
    def __init__(self, output, w, h, frame_ms, suite):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.w, self.h, self.ms, self.suite = w,h,frame_ms,suite
        self.frames, self.segments, self.parts = 0, [], []

    def append(self, name, frames, *, phase, kind, direction=None, speed=0., source='', heading=None):
        images = list(frames)
        if not images or any(len(f) != self.w*self.h for f in images):
            raise ValueError('invalid image sequence')
        first = self.frames
        self.frames += len(images)
        self.parts.append(images)
        self.segments.append(dict(segment_index=len(self.segments)+1, name=name, phase=phase,
            kind=kind, direction_slot_deg=direction if direction is not None else 0,
            expected_direction_deg=direction, speed_px_per_frame=speed, is_blank=0,
            first_frame=first+1, last_frame=self.frames, start_ms=first*self.ms,
            end_ms=self.frames*self.ms, source_file=source, expected_heading_deg=heading))

    def finish(self, extra=None):
        path = self.output/'all_stimuli.dat'
        write_stimulus(path,self.w,self.h,(frame for images in self.parts for frame in images))
        with (self.output/'segments.csv').open('x',encoding='utf-8',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(self.segments[0]))
            writer.writeheader()
            writer.writerows(self.segments)
        meta = dict(schema='visual-stimulus-sequence-v1',suite_type=self.suite,width=self.w,height=self.h,
            frame_ms=self.ms,coordinates='x-right_y-down; clockwise from right',
            runs=dict(stimulus='all_stimuli.dat',stimulus_sha256=file_sha256(path),
                segments='segments.csv',frames=self.frames,duration_ms=self.frames*self.ms,spikes=None),
            boundaries='Run whole DAT continuously in V1; never treat independent source cases as one physical trajectory.',
            analysis='Use per-segment labels. Static texture is not a black screen. Exclude startup/scene cuts explicitly.')
        meta.update(extra or {})
        (self.output/'manifest.json').write_text(json.dumps(meta,indent=2,allow_nan=False),encoding='utf-8')
        return meta


def checked_sources(folder, meta):
    for run in meta['runs']:
        path = folder/run['stimulus']
        info = inspect_stimulus(path)
        if (info['width'],info['height'],info['frames'],info['channels']) != (meta['width'],meta['height'],meta['frames'],1):
            raise ValueError(f'{path}: source grid/frame mismatch')
        if file_sha256(path) != run['stimulus_sha256']:
            raise ValueError(f'{path}: source checksum mismatch')


def prepare(texture_suite, selfmotion_suite, output):
    texture_suite, selfmotion_suite, output = map(Path,(texture_suite,selfmotion_suite,output))
    texture = json.loads((texture_suite/'manifest.json').read_text(encoding='utf-8-sig'))
    motion = json.loads((selfmotion_suite/'manifest.json').read_text(encoding='utf-8-sig'))
    if texture.get('schema') != 'v1-direction-calibration-v1' or texture.get('kind') != 'texture':
        raise ValueError('expected original texture calibration suite')
    if motion.get('schema') != 'selfmotion-stimulus-v1':
        raise ValueError('expected original self-motion suite')
    checked_sources(texture_suite, texture)
    checked_sources(selfmotion_suite, motion)
    by_angle = {r['expected_direction_deg']:r for r in texture['runs']}
    if set(by_angle) != set(range(0,360,45)):
        raise ValueError('requires all eight texture directions')
    by_name = {r['name']:r for r in motion['runs']}
    if any(name not in by_name for name in BASELINE):
        raise ValueError('missing self-motion baseline case')
    output.mkdir(parents=True,exist_ok=False)
    w,h,ms,n = (texture[k] for k in ('width','height','frame_ms','frames'))
    hold = max(1,2000//ms)
    static = max(1,4000//ms)
    b = Builder(output/'texture',w,h,ms,'texture')
    first = next(read_images(texture_suite/by_angle[0]['stimulus']))
    b.append('static_initial',[first]*static,phase='static_control',kind='static')
    source_hashes=[]
    for direction in range(0,360,45):
        run = by_angle[direction]
        images = list(read_images(texture_suite/run['stimulus']))
        b.append(run['name'],images,phase='original_eight_directions',kind='translation',
                 direction=direction,speed=texture['speed_px_per_frame'],source=run['stimulus'])
        b.append(f'static_after_{direction:03d}',[images[-1]]*hold,
                 phase='static_after_motion',kind='static',source=run['stimulus'])
        source_hashes.append(dict(name=run['name'],sha256=run['stimulus_sha256']))
    # Reset here is explicit and separated by a stationary interval, NOT a claimed smooth switch.
    b.append('static_before_continuous',[first]*hold,phase='scene_reset_control',kind='static')
    continuous = list(continuous_texture(w,h,n,texture['speed_px_per_frame'],texture['seed']))
    for i,direction in enumerate(range(0,360,45)):
        b.append(f'continuous_d{direction:03d}',[f for _,f in continuous[i*n:(i+1)*n]],
                 phase='continuous_direction_switch',kind='translation',direction=direction,
                 speed=texture['speed_px_per_frame'])
    b.append('static_final',[continuous[-1][1]]*static,phase='static_after_continuous',kind='static')
    texture_meta = b.finish(dict(seed=texture['seed'],source_runs=source_hashes,
        notes=['Original 8 direction payloads are unchanged; boundaries may reset spatial phase.',
               'Continuous block carries one texture position across directions, with no blank or phase reset.',
               'First/last windows at direction boundaries are transition windows; inspect separately.',
               'Stationary controls hold actual texture pixels; they are not black.']))
    w,h,ms = (motion[k] for k in ('width','height','frame_ms'))
    b = Builder(output/'selfmotion',w,h,ms,'selfmotion')
    selected=[]
    with (b.output/'truth.jsonl').open('x',encoding='utf-8') as truth_stream:
        for name in BASELINE:
            run=by_name[name]
            first_frame=b.frames
            heading=run['motion']['heading_deg'] if run['motion']['mode']=='straight' else None
            b.append(name,read_images(selfmotion_suite/run['stimulus']),phase='selfmotion_baseline',
                     kind='static' if run['motion']['mode']=='static' else 'selfmotion',
                     source=run['stimulus'],heading=heading)
            count=0
            with (selfmotion_suite/run['truth']).open(encoding='utf-8') as source:
                for i,line in enumerate(source):
                    record=json.loads(line)
                    truth_stream.write(json.dumps(dict(combined_frame=first_frame+i+1,
                        combined_time_s=(first_frame+i)*ms/1000,condition=name,scene_cut=i==0,
                        coordinate_scope='source condition local coordinates; not a continuous camera trajectory',
                        source_truth=record),allow_nan=False)+'\n')
                    count+=1
            if count != motion['frames']:
                raise ValueError('source truth frame count mismatch')
            selected.append(dict(name=name,source_sha256=run['stimulus_sha256'],scene=run['scene'],motion=run['motion']))
    shutil.copyfile(selfmotion_suite/'intrinsics.json',b.output/'intrinsics.json')
    motion_meta=b.finish(dict(source_runs=selected,intrinsics_px=motion['intrinsics_px'],
        truth='truth.jsonl',camera=motion['camera'],
        notes=['Only five current baseline conditions, not all 14 unsupported expansion cases.',
               'Independent scene/pose resets occur at each condition boundary. No inter-case camera flow truth.',
               'Do not reset V1 within continuous acquisition; segment/reset MSTd offline at case boundaries.',
               'Exclude initial transients per case; do not judge scene cuts as real heading changes.']))
    return {'texture':texture_meta['runs'],'selfmotion':motion_meta['runs']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--texture-suite',default='outputs/calibration_texture_01')
    p.add_argument('--selfmotion-suite',default='outputs/selfmotion_01')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    print(json.dumps(prepare(a.texture_suite,a.selfmotion_suite,a.output),indent=2))


if __name__=='__main__':
    main()
