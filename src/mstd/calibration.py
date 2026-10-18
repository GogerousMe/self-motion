"""Known-direction visual stimuli and spike-based projection calibration.

No V1 simulator is bundled: generated DATs must run through the actual C++ V1.
Image angles are clockwise from right (x-right, y-down).
"""
import hashlib
import json
from math import atan2, cos, degrees, floor, hypot, isfinite, pi, sin
from pathlib import Path
import random
import struct

from .carlsim import iter_bins, read_header
from .data import V1Spec
from .projection import load_projection, project_counts

STIM_SIGNATURE = 293390619


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_stimulus(path):
    path = Path(path)
    with path.open('rb') as stream:
        raw = stream.read(25)
    if len(raw) != 25:
        raise ValueError('truncated VisualStimulus header')
    signature,version,kind,channels,w,h,frames = struct.unpack('<ifiBiii',raw)
    if signature != STIM_SIGNATURE or version != 1 or kind not in range(7):
        raise ValueError('unsupported VisualStimulus DAT')
    if channels not in (1,3) or min(w,h,frames) <= 0:
        raise ValueError('invalid stimulus dimensions')
    if path.stat().st_size != 25+w*h*channels*frames:
        raise ValueError('stimulus binary size mismatch')
    return {'width':w,'height':h,'frames':frames,'channels':channels,'type':kind}


def write_stimulus(path, width, height, frames):
    """VisualStimulus v1 grayscale picture DAT: 25-byte header + TYX uint8."""
    if type(width) is not int or type(height) is not int or min(width, height) < 2:
        raise ValueError('integer width and height >=2 required')
    count = 0
    with Path(path).open('xb') as stream:
        stream.write(struct.pack('<ifiBiii', STIM_SIGNATURE, 1.0, 4, 1, width, height, 0))
        for frame in frames:
            if len(frame) != width*height:
                raise ValueError('stimulus frame dimensions mismatch')
            stream.write(bytes(frame))
            count += 1
        if count == 0:
            raise ValueError('empty stimulus')
        stream.seek(21)
        stream.write(struct.pack('<i', count))
    return count


def stimulus_frames(width, height, count, angle_deg, speed, period=12.0, kind='grating', seed=0):
    if type(count) is not int or count <= 0:
        raise ValueError('positive integer frame count required')
    if not all(isfinite(v) for v in (angle_deg, speed, period)) or speed <= 0 or period <= 2:
        raise ValueError('positive speed and spatial period >2 required')
    if kind not in ('grating', 'texture'):
        raise ValueError('kind must be grating or texture')
    if type(width) is not int or type(height) is not int or min(width, height) < 2:
        raise ValueError('integer width and height >=2 required')
    dx, dy = cos(angle_deg*pi/180), sin(angle_deg*pi/180)
    rng = random.Random(seed)
    texture = [230 if rng.random() < .2 else 25 for _ in range(width*height)]
    for t in range(count):
        frame = bytearray()
        for y in range(height):
            for x in range(width):
                if kind == 'grating':
                    value = 127.5+100*cos(2*pi*(x*dx+y*dy-speed*t)/period)
                else:
                    # Periodic translation of one fixed random texture, bilinear samples.
                    sx, sy = x-speed*t*dx, y-speed*t*dy
                    ix, iy = floor(sx), floor(sy)
                    ax, ay = sx-ix, sy-iy
                    value = sum(texture[((iy+oy)%height)*width+(ix+ox)%width]*wx*wy
                                for ox,wx in ((0,1-ax),(1,ax)) for oy,wy in ((0,1-ay),(1,ay)))
                # Stable half-up quantization at cosine zero crossings: tiny
                # trig roundoff must not create direction-dependent 1-LSB flicker.
                frame.append(max(0, min(255, floor(value+.5+1e-9))))
        yield bytes(frame)


def generate_suite(output, width=80, height=80, frames=80, frame_ms=50,
                   warmup_frames=20, speed=1.5, kind='grating', seed=0):
    if type(frame_ms) is not int or frame_ms <= 0 or type(warmup_frames) is not int:
        raise ValueError('integer frame timing required')
    if type(frames) is not int or not 0 <= warmup_frames < frames:
        raise ValueError('warmup must leave evaluation frames')
    # Preflight generator arguments before creating the output directory.
    sample = stimulus_frames(width, height, frames, 0, speed, kind=kind, seed=seed)
    first = next(sample)
    sample.close()
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    runs = []
    for angle in range(0, 360, 45):
        name = f'{kind}_d{angle:03d}'
        path = directory/f'{name}.dat'
        write_stimulus(path, width, height,
                       stimulus_frames(width, height, frames, angle, speed, kind=kind, seed=seed))
        runs.append({'name': name, 'expected_direction_deg': angle,
                     'stimulus': path.name, 'stimulus_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                     'spikes': None})
    manifest = {'schema': 'v1-direction-calibration-v1', 'width': width, 'height': height,
                'frames': frames, 'frame_ms': frame_ms, 'duration_ms': frames*frame_ms,
                'warmup_ms': warmup_frames*frame_ms, 'speed_px_per_frame': speed,
                'kind': kind, 'seed': seed, 'coordinates': 'x-right_y-down',
                'directions': 'degrees clockwise from right', 'runs': runs,
                'instructions': 'Run each DAT through a reset C++ V1 with frame_ms timing. '
                                'Set spikes to matching SpikeMonitor files. Do not use stimulus DAT as spikes.'}
    (directory/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (directory/'preview.pgm').write_bytes(f'P5\n{width} {height}\n255\n'.encode()+first)
    return {'output': str(directory), 'runs': len(runs), 'manifest': str(directory/'manifest.json')}


def analyze_calibration(manifest_path, projection_path, projection_name='motionProj1', border=4,
                        tolerance_deg=22.5, min_resultant=.1):
    manifest_path = Path(manifest_path)
    meta = json.loads(manifest_path.read_text(encoding='utf-8'))
    if meta.get('schema') != 'v1-direction-calibration-v1' or meta.get('coordinates') != 'x-right_y-down':
        raise ValueError('unsupported calibration manifest')
    w, h, duration, frame_ms, warmup = (meta[k] for k in ('width','height','duration_ms','frame_ms','warmup_ms'))
    if any(type(v) is not int for v in (w,h,duration,frame_ms,warmup,border)):
        raise ValueError('integer grid, timing and border required')
    if min(w,h) < 2 or duration <= 0 or frame_ms <= 0 or duration % frame_ms or warmup % frame_ms:
        raise ValueError('positive duration and aligned frame timing required')
    if type(meta.get('frames')) is not int or meta['frames']*frame_ms != duration:
        raise ValueError('manifest frame count does not match duration')
    if not 0 <= warmup < duration or not 0 <= border < min(w,h)/2:
        raise ValueError('invalid warmup or ROI border')
    if not isfinite(tolerance_deg) or not 0 <= tolerance_deg <= 180:
        raise ValueError('invalid angle tolerance')
    if not isfinite(min_resultant) or not 0 <= min_resultant <= 1:
        raise ValueError('invalid resultant threshold')
    projection = load_projection(projection_path, projection_name)
    spec = V1Spec(w,h,projection.directions_rad,1,1,0,0)  # Intrinsics unused for image-direction calibration.
    roi = [y*w+x for y in range(border,h-border) for x in range(border,w-border)]
    n = w*h
    rows = []
    names = set()
    runs = meta['runs']
    if not runs:
        raise ValueError('calibration manifest has no runs')
    for run in runs:
        name, expected = run['name'], run['expected_direction_deg']
        if name in names or not isfinite(expected):
            raise ValueError('duplicate run name or invalid expected direction')
        names.add(name)
        if not run.get('spikes'):
            rows.append({'name':name,'status':'pending','expected_direction_deg':expected})
            continue
        stimulus = manifest_path.parent/run['stimulus']
        info = inspect_stimulus(stimulus)
        if (info['width'],info['height'],info['frames']) != (w,h,meta['frames']):
            raise ValueError(f'{name}: stimulus dimensions do not match manifest')
        if hashlib.sha256(stimulus.read_bytes()).hexdigest() != run['stimulus_sha256']:
            raise ValueError(f'{name}: stimulus checksum mismatch')
        path = manifest_path.parent/run['spikes']
        recorded_hash = run.get('acquisition',{}).get('spikes_sha256')
        if recorded_hash and file_sha256(path) != recorded_hash:
            raise ValueError(f'{name}: spike checksum mismatch')
        header = read_header(path)
        if (header.width,header.height,header.depth) != (w,h,84):
            raise ValueError(f'{name}: spike grid does not match manifest')
        direction_totals = [0.0]*8
        filter_counts = [0]*84
        # Fully consume events, including warmup, to validate complete recording.
        for start, width, counts in iter_bins(path,duration,frame_ms):
            if start < warmup:
                continue
            rates = project_counts(spec,counts,width,projection)
            for d in range(8):
                direction_totals[d] += sum(rates[d*n+p] for p in roi)*width
            for c in range(84):
                filter_counts[c] += sum(counts[c*n+p] for p in roi)
        mean_rates = [value/((duration-warmup)*len(roi)) for value in direction_totals]
        total = sum(mean_rates)
        vx = sum(r*cos(a) for r,a in zip(mean_rates,projection.directions_rad))
        vy = sum(r*sin(a) for r,a in zip(mean_rates,projection.directions_rad))
        resultant = hypot(vx,vy)/total if total else 0.0
        responsive = total > 1e-12 and resultant >= min_resultant
        order = sorted(range(8),key=mean_rates.__getitem__,reverse=True)
        winner = degrees(projection.directions_rad[order[0]])%360 if responsive else None
        vector_angle = degrees(atan2(vy,vx))%360 if responsive else None
        error = abs((winner-expected+180)%360-180) if responsive else None
        rows.append({'name':name,'status':'measured','expected_direction_deg':expected,
                     'winner_direction_deg':winner,'population_direction_deg':vector_angle,
                     'winner_error_deg':error,'resultant':resultant,'responsive':responsive,
                     'within_tolerance':responsive and error <= tolerance_deg,
                     'direction_mean_hz':mean_rates,
                     'filter_mean_hz': [c*1000/((duration-warmup)*len(roi)) for c in filter_counts],
                     'events':header.event_count})
    measured = [r for r in rows if r['status']=='measured']
    eight_directions = {round(r['expected_direction_deg']%360,6) for r in rows} == set(range(0,360,45))
    return {'schema':'v1-direction-calibration-report-v1','projection_name':projection_name,
            'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            'projection_config_sha256':hashlib.sha256(Path(projection_path).read_bytes()).hexdigest(),
            'border':border,'warmup_ms':warmup,'tolerance_deg':tolerance_deg,'min_resultant':min_resultant,
            'measured_runs':len(measured),'pending_runs':len(rows)-len(measured),
            'calibration_passed':eight_directions and len(measured)==len(rows) and all(r['within_tolerance'] for r in measured),
            'runs':rows,'limitations':['Thresholds are engineering criteria, not calibrated confidence.',
                                       'Grating aperture ambiguity: repeat with translated texture.',
                                       'Response/stimulus pairing must be established by the acquisition process.']}
