"""Two-row stimulus / eight spatial MT activity video. Stdlib + ffmpeg only.

First run reads raw V1 spikes and caches real normalized MT direction maps.
Later --cache runs render the maps without rereading the large spike recording.
"""
import argparse
from array import array
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import sys
from time import perf_counter
import zlib

from mstd.calibration import inspect_stimulus, file_sha256
from mstd.carlsim import iter_bins, read_header
from mstd.data import V1Frame, V1Spec
from mstd.mt import encode_mt
from mstd.projection import load_projection, project_counts


# Portable bitmap lettering; no GUI, Pillow or system fonts needed on Linux.
FONT = dict(zip('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', [
 '01110/10001/10011/10101/11001/10001/01110','00100/01100/00100/00100/00100/00100/01110',
 '01110/10001/00001/00010/00100/01000/11111','11110/00001/00001/01110/00001/00001/11110',
 '00010/00110/01010/10010/11111/00010/00010','11111/10000/10000/11110/00001/00001/11110',
 '01110/10000/10000/11110/10001/10001/01110','11111/00001/00010/00100/01000/01000/01000',
 '01110/10001/10001/01110/10001/10001/01110','01110/10001/10001/01111/00001/00001/01110',
 '01110/10001/10001/11111/10001/10001/10001','11110/10001/10001/11110/10001/10001/11110',
 '01111/10000/10000/10000/10000/10000/01111','11110/10001/10001/10001/10001/10001/11110',
 '11111/10000/10000/11110/10000/10000/11111','11111/10000/10000/11110/10000/10000/10000',
 '01111/10000/10000/10111/10001/10001/01111','10001/10001/10001/11111/10001/10001/10001',
 '01110/00100/00100/00100/00100/00100/01110','00001/00001/00001/00001/10001/10001/01110',
 '10001/10010/10100/11000/10100/10010/10001','10000/10000/10000/10000/10000/10000/11111',
 '10001/11011/10101/10101/10001/10001/10001','10001/11001/10101/10011/10001/10001/10001',
 '01110/10001/10001/10001/10001/10001/01110','11110/10001/10001/11110/10000/10000/10000',
 '01110/10001/10001/10001/10101/10010/01101','11110/10001/10001/11110/10100/10010/10001',
 '01111/10000/10000/01110/00001/00001/11110','11111/00100/00100/00100/00100/00100/00100',
 '10001/10001/10001/10001/10001/10001/01110','10001/10001/10001/10001/10001/01010/00100',
 '10001/10001/10001/10101/10101/10101/01010','10001/10001/01010/00100/01010/10001/10001',
 '10001/10001/01010/00100/00100/00100/00100','11111/00001/00010/00100/01000/10000/11111']))
FONT.update({'.':'00000/00000/00000/00000/00000/00110/00110',
             ':':'00000/00110/00110/00000/00110/00110/00000',
             '-':'00000/00000/00000/11111/00000/00000/00000',
             '/':'00001/00010/00010/00100/01000/01000/10000',
             '=':'00000/11111/00000/11111/00000/00000/00000'})
NAMES = ['RIGHT', 'RIGHT-DOWN', 'DOWN', 'LEFT-DOWN', 'LEFT', 'LEFT-UP', 'UP', 'RIGHT-UP']
W, H = 1440, 640


def rect(canvas, x, y, w, h, color):
    for yy in range(max(0, y), min(H, y+h)):
        start, end = max(0, x), min(W, x+w)
        if end > start:
            canvas[(yy*W+start)*3:(yy*W+end)*3] = bytes(color)*(end-start)


def text(canvas, x, y, message, scale=2, color=(230, 235, 245)):
    for char in message.upper():
        for yy, row in enumerate(FONT.get(char, '00000/'*6+'00000').split('/')):
            for xx, bit in enumerate(row):
                if bit == '1':
                    rect(canvas, x+xx*scale, y+yy*scale, scale, scale, color)
        x += 6*scale


def arrow(canvas, x, y, angle, length=30, color=(255, 205, 70)):
    a = math.radians(angle)
    endx, endy = x+length*math.cos(a), y+length*math.sin(a)
    for x0, y0, x1, y1 in [(x, y, endx, endy)] + [
            (endx, endy, endx-12*math.cos(a+b), endy-12*math.sin(a+b)) for b in (-.55, .55)]:
        steps = int(max(abs(x1-x0), abs(y1-y0)))+1
        for i in range(steps+1):
            rect(canvas, round(x0+(x1-x0)*i/steps)-1, round(y0+(y1-y0)*i/steps)-1, 3, 3, color)


def color(value, maximum, palette='hot'):
    v = max(0., min(1., value/maximum))
    if palette == 'jet':
        return tuple(round(255*max(0., min(1., 1.5-abs(4*v-k)))) for k in (3, 2, 1))
    return (round(255*min(1., 3*v)), round(255*max(0., min(1., 3*v-1))),
            round(255*max(0., 3*v-2)))


def image(canvas, x, y, size, values, width, height, vmax=None, palette='hot'):
    # Nearest-neighbor scales preserve the original spatial samples.
    previous_y, line = None, None
    for yy in range(size):
        source_y = yy*height//size
        if source_y != previous_y:
            colors = []
            for sx in range(width):
                v = values[source_y*width+sx]
                colors.append(bytes((v, v, v) if vmax is None else color(v, vmax, palette)))
            line = b''.join(colors[xx*width//size] for xx in range(size))
            previous_y = source_y
        offset = ((y+yy)*W+x)*3
        canvas[offset:offset+size*3] = line


def png(path, rgb, width=W, height=H):
    def chunk(kind, data):
        return struct.pack('>I', len(data))+kind+data+struct.pack('>I', zlib.crc32(kind+data))
    data = b''.join(b'\x00'+rgb[y*width*3:(y+1)*width*3] for y in range(height))
    Path(path).write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
                           +chunk(b'IDAT', zlib.compress(data))+chunk(b'IEND', b''))


def render(stimulus, channels, spec, segment, frame_index, total, frame_ms, vmax,
           response='normalized', palette='hot'):
    c = bytearray(bytes((15, 21, 33))*(W*H))
    text(c, 20, 14, 'INPUT STIMULUS AND MT DIRECTION ACTIVITY', 3)
    image(c, 20, 56, 240, stimulus, spec.width, spec.height)
    moving = not segment['is_blank'] and segment['speed_px_per_frame'] > 0
    expected = int(segment['direction_slot_deg']) % 360 if moving else None
    text(c, 295, 65, f'FRAME {frame_index+1}/{total}  SEGMENT {segment["segment_index"]}', 3)
    text(c, 295, 103, f'RESPONSE WINDOW {frame_index*frame_ms/1000:.2f}-{(frame_index+1)*frame_ms/1000:.2f} S')
    label = NAMES[(expected//45)%8] + f' {expected} DEG' if moving else ('BLANK - NO DIRECTION' if segment['is_blank'] else 'STATIC - NO DIRECTION')
    text(c, 295, 141, 'INPUT: '+label, 3, (255, 205, 70))
    if moving:
        arrow(c, 1230, 153, expected, 55)
    text(c, 295, 183, f'SPEED {segment["speed_px_per_frame"]:g} PX/FRAME')
    titles = {'normalized': 'ACTIVITY: POOLED AND NORMALIZED MT CHANNELS - NOT HZ',
              'projected-hz': 'V1 TO 8-DIRECTION PROJECTED RATE - HZ - NOT MT SPIKES',
              'pooled-hz': 'LOCAL MT POOLED INPUT RATE - HZ - BEFORE NORMALIZATION',
              'mt-spikes-hz': 'ACTUAL MT SPIKE RATE - HZ = COUNT / WINDOW SECONDS'}
    text(c, 295, 216, titles[response])
    text(c, 295, 246, 'YELLOW: EXPECTED   CYAN: STRONGEST CHANNEL MEAN', color=(140, 215, 235))
    stage_titles = {'projected-hz': 'STAGE 1: PROJECTED RATE - HZ',
                    'pooled-hz': 'STAGE 2: LOCAL POOLED RATE - HZ',
                    'normalized': 'STAGE 3: NORMALIZED MT ACTIVITY',
                    'mt-spikes-hz': 'MT SPIKE RATE - HZ'}
    text(c, 20, 315, stage_titles[response]
         + ' - CLOCKWISE FROM RIGHT', 2)
    n = spec.width*spec.height
    means = [sum(ch)/n for ch in channels]
    winner = max(range(8), key=means.__getitem__) if max(means) > 1e-8 else None
    for d, ch in enumerate(channels):
        x = 20+d*177
        text(c, x, 345, f'{d*45} DEG {NAMES[d]}', 1)
        if expected == d*45:
            rect(c, x, 362, 160, 4, (255, 205, 70))
        if winner == d:
            rect(c, x, 530, 160, 4, (60, 225, 245))
        image(c, x, 368, 160, ch, spec.width, spec.height, vmax, palette)
        text(c, x, 544, f'MEAN {means[d]:.3f}')
    text(c, 20, 588, 'FIXED COLOR SCALE: 0', 2)
    for i in range(240):
        rect(c, 290+i, 585, 1, 20, color(vmax*i/239, vmax, palette))
    unit = 'HZ' if response != 'normalized' else 'NORMALIZED'
    text(c, 548, 588, f'{vmax:g} {unit}  SAME SCALE FOR ALL FRAMES')
    return c


def rate_channels(counts, spec, width_ms):
    """Actual SpikeMonitor count-to-Hz conversion, plane-major DYX layout."""
    if width_ms <= 0 or len(counts) != spec.frame_size:
        raise ValueError('expected eight direction planes and positive window duration')
    n = spec.width*spec.height
    return tuple(tuple(counts[d*n+p]*1000/width_ms for p in range(n)) for d in range(8))


def load_segments(meta, parent):
    run, ms = meta['runs'], meta['frame_ms']
    with (parent/run['segments']).open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    end = 0
    for s in rows:
        for key in ('segment_index', 'first_frame', 'last_frame', 'start_ms', 'end_ms', 'is_blank'):
            s[key] = int(s[key])
        for key in ('direction_slot_deg', 'speed_px_per_frame'):
            s[key] = float(s[key])
        if (s['start_ms'] != end or s['end_ms'] <= end or s['is_blank'] not in (0, 1)
                or s['start_ms'] != (s['first_frame']-1)*ms or s['end_ms'] != s['last_frame']*ms
                or not math.isfinite(s['speed_px_per_frame']) or s['speed_px_per_frame'] < 0
                or not math.isfinite(s['direction_slot_deg'])):
            raise ValueError('invalid segment timing or labels')
        end = s['end_ms']
    if not rows or end != run['duration_ms']:
        raise ValueError('segments do not cover duration')
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--stimulus', help='actual input image DAT path; must match manifest dimensions and SHA256')
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--spikes', help='raw V1 SpikeMonitor DAT; computes and caches MT maps')
    source.add_argument('--mt-spikes', help='actual eight-plane MT SpikeMonitor DAT; count-to-Hz heatmaps')
    source.add_argument('--cache', help='mt_activity.json from an earlier successful visualization')
    p.add_argument('--response', choices=['normalized', 'projected-hz'],
                   help='for --spikes: normalized MT activity (default) or pre-MT projected Hz')
    p.add_argument('--mt-directions-deg', type=float, nargs=8,
                   help='required with --mt-spikes: physical angle of each stored plane, x-right y-down')
    p.add_argument('--output', required=True, help='new output directory')
    p.add_argument('--projection-config', default='configs/v1_projection.json')
    p.add_argument('--projection', default='motionProj1')
    p.add_argument('--radius', type=int, default=1)
    p.add_argument('--max-frames', type=int, help='prefix preview; omit for all frames')
    p.add_argument('--vmax', type=float, help='fixed heatmap maximum: default 1 normalized, 100 Hz for rates')
    p.add_argument('--colormap', choices=['hot', 'jet'], default='hot')
    p.add_argument('--playback-rate', type=float, default=1.)
    p.add_argument('--progress-every', type=int, default=20)
    p.add_argument('--ffmpeg', default='ffmpeg')
    a = p.parse_args(argv)
    if (a.radius < 0 or a.progress_every < 0 or a.vmax is not None and (not math.isfinite(a.vmax) or a.vmax <= 0)
            or not math.isfinite(a.playback_rate) or a.playback_rate <= 0
            or a.max_frames is not None and a.max_frames <= 0):
        p.error('invalid radius, frame count, color scale, playback or progress setting')
    if a.mt_spikes and (a.mt_directions_deg is None or a.response is not None):
        p.error('--mt-spikes requires --mt-directions-deg; do not supply --response')
    if a.mt_directions_deg is not None and not a.mt_spikes:
        p.error('--mt-directions-deg is only used with --mt-spikes')
    if a.cache and a.response is not None:
        p.error('cache units are fixed at creation; --response cannot change them')
    ffmpeg = shutil.which(a.ffmpeg)
    if not ffmpeg:
        p.error('ffmpeg not found; install ffmpeg or pass --ffmpeg /path/to/ffmpeg')
    manifest = Path(a.manifest)
    meta = json.loads(manifest.read_text(encoding='utf-8-sig'))
    direction_schema = (meta.get('schema') == 'matlab-sequential-binary-gratings-v2'
                        or meta.get('schema') == 'visual-stimulus-sequence-v1' and meta.get('suite_type') == 'texture')
    if (not direction_schema
            or meta.get('coordinates') != 'x-right_y-down; clockwise from right'):
        raise ValueError('requires continuous MATLAB v2 manifest with image coordinates')
    run, ms = meta['runs'], meta['frame_ms']
    stimulus = Path(a.stimulus) if a.stimulus else manifest.parent/run['stimulus']
    info = inspect_stimulus(stimulus)
    if (info['width'], info['height'], info['frames'], info['channels']) != (meta['width'], meta['height'], run['frames'], 1):
        raise ValueError('stimulus/manifest shape mismatch')
    if ms <= 0 or run['duration_ms'] != run['frames']*ms or file_sha256(stimulus) != run['stimulus_sha256']:
        raise ValueError('stimulus checksum or duration mismatch')
    segments = load_segments(meta, manifest.parent)
    proj = load_projection(a.projection_config, a.projection) if a.spikes else None
    cached = json.loads(Path(a.cache).read_text(encoding='utf-8')) if a.cache else None
    if cached and (cached['schema'] != 'mt-direction-map-cache-v1'
            or cached['manifest_sha256'] != file_sha256(manifest)
            or cached['segments_sha256'] != file_sha256(manifest.parent/run['segments'])):
        raise ValueError('cache does not match manifest')
    response = cached.get('response_kind', 'normalized') if cached else ('mt-spikes-hz' if a.mt_spikes else a.response or 'normalized')
    a.vmax = a.vmax if a.vmax is not None else (1. if response == 'normalized' else 100.)
    source_spikes = a.mt_spikes or a.spikes
    directions = (tuple(math.radians(v) for v in a.mt_directions_deg) if a.mt_spikes else
                  proj.directions_rad if proj else tuple(cached['directions_rad']))
    spec = V1Spec(meta['width'], meta['height'], directions, 1, 1, 0, 0)
    order = [min(range(8), key=lambda j: abs((math.degrees(directions[j])-d*45+180)%360-180)) for d in range(8)]
    if len(set(order)) != 8 or any(abs((math.degrees(directions[j])-d*45+180)%360-180)>1e-5 for d,j in enumerate(order)):
        raise ValueError('visualization requires directions at multiples of 45 degrees')
    total = min(run['frames'], a.max_frames or run['frames'])
    if cached:
        total = min(total, cached['frames'])
        if (cached['width'], cached['height']) != (spec.width, spec.height):
            raise ValueError('cache grid mismatch')
        cache_path = Path(a.cache).parent/cached['data_file']
        if cache_path.stat().st_size != cached['frames']*spec.frame_size*4:
            raise ValueError('cache size mismatch')
        if file_sha256(cache_path) != cached['data_sha256']:
            raise ValueError('cache checksum mismatch')
    else:
        header = read_header(source_spikes)
        expected_depth = 8 if a.mt_spikes else 84
        if (header.width, header.height, header.depth) != (spec.width, spec.height, expected_depth):
            raise ValueError(f'spike grid must match stimulus W x H x {expected_depth}')
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', f'{W}x{H}', '-r', str(1000/ms*a.playback_rate), '-i', '-', '-an',
               '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(out/'mt_directions.mp4')]
    start, seg_index, count = perf_counter(), 0, 0
    digest = hashlib.sha256()
    cache_stream = cache_path.open('rb') if cached else (out/'mt_activity.f32').open('xb')
    bins = iter_bins(source_spikes, run['duration_ms'], ms) if source_spikes else None
    print(f'[start] Rendering {total} frames; {response}, fixed color range 0..{a.vmax}', flush=True)
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        with stimulus.open('rb') as f, cache_stream:
            f.seek(25)
            for i in range(total):
                pixels = f.read(spec.width*spec.height)
                while i*ms >= segments[seg_index]['end_ms']:
                    seg_index += 1
                if cached:
                    packed = cache_stream.read(spec.frame_size*4)
                    values = array('f')
                    values.frombytes(packed)
                    if sys.byteorder != 'little':
                        values.byteswap()
                    channels = tuple(values[d*spec.width*spec.height:(d+1)*spec.width*spec.height] for d in range(8))
                else:
                    t, width, counts = next(bins)
                    if t != i*ms or width != ms:
                        raise ValueError('spike/stimulus window mismatch')
                    if a.mt_spikes:
                        channels = rate_channels(counts, spec, width)
                    else:
                        rates = project_counts(spec, counts, width, proj)
                        if response == 'projected-hz':
                            n = spec.width*spec.height
                            channels = tuple(rates[d*n:(d+1)*n] for d in range(8))
                        else:
                            channels = encode_mt(spec, V1Frame(t/1000, rates), a.radius, return_channels=True).direction_activity
                    values = array('f', (v for ch in channels for v in ch))
                    if sys.byteorder != 'little':
                        values.byteswap()
                    packed = values.tobytes()
                    digest.update(packed)
                    cache_stream.write(packed)
                canvas = render(pixels, [channels[j] for j in order], spec, segments[seg_index], i, total,
                                ms, a.vmax, response, a.colormap)
                process.stdin.write(canvas)
                count += 1
                segment = segments[seg_index]
                preview_offset = min(1000, ((segment['end_ms']-segment['start_ms'])//(2*ms))*ms)
                if i == 0 or (i*ms == segment['start_ms']+preview_offset):
                    png(out/f'preview_segment_{segments[seg_index]["segment_index"]:03d}.png', canvas)
                if a.progress_every and (count == 1 or count % a.progress_every == 0 or count == total):
                    elapsed = perf_counter()-start
                    print(f'[progress] {count}/{total}, elapsed={elapsed:.1f}s, remaining~{elapsed/count*(total-count):.1f}s', flush=True)
            # Exhaust the final generator boundary to validate the full input, including out-of-range tail events.
            if bins is not None and total == run['frames']:
                if next(bins, None) is not None:
                    raise ValueError('unexpected extra bin')
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError('ffmpeg failed; see stderr')
    except BaseException:
        if process.poll() is None:
            process.kill()
        process.wait()
        raise
    finally:
        if bins is not None:
            bins.close()
    if not cached:
        cache_meta = dict(schema='mt-direction-map-cache-v1', width=spec.width, height=spec.height,
            frames=count, frame_ms=ms, data_file='mt_activity.f32', data_sha256=digest.hexdigest(),
            encoding='little-endian float32 DYX per frame', directions_rad=directions,
            manifest_sha256=file_sha256(manifest), stimulus_sha256=run['stimulus_sha256'],
            segments_sha256=file_sha256(manifest.parent/run['segments']),
            response_kind=response, units='normalized activity' if response == 'normalized' else 'Hz',
            projection_config_sha256=file_sha256(a.projection_config) if a.spikes else None,
            projection_name=a.projection if a.spikes else None,
            radius=a.radius if response == 'normalized' else None, spikes_source=str(Path(source_spikes).resolve()),
            spikes_bytes=Path(source_spikes).stat().st_size, spikes_events=header.event_count,
            full_event_validation=total == run['frames'])
        (out/'mt_activity.json').write_text(json.dumps(cache_meta, indent=2), encoding='utf-8')
    (out/'video_config.json').write_text(json.dumps(dict(frames=count, frame_ms=ms,
        stimulus_file=str(stimulus.resolve()),
        playback_rate=a.playback_rate, color_max=a.vmax, colormap=a.colormap,
        response_kind=response, output_direction_order_deg=list(range(0,360,45)),
        source_cache=str(Path(a.cache).resolve()) if cached else 'mt_activity.json',
        normalized_activity_not_hz=response == 'normalized', response_time='mean over displayed interval; available at interval end',
        stimulus='frame at interval start', complete_manifest_rendered=count == run['frames']), indent=2), encoding='utf-8')
    print(f'[done] {out}/mt_directions.mp4', flush=True)


if __name__ == '__main__':
    main()
