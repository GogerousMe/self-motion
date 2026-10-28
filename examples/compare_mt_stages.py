"""One V1 pass -> input + projected / pooled / normalized eight-direction rows.

Requires Python >=3.10 and ffmpeg. MT model outputs are not simulated spikes.
Stores all three stages for later --cache rendering without rereading V1.
"""
import argparse
from array import array
import csv
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter

from mstd.calibration import file_sha256, inspect_stimulus
from mstd.carlsim import iter_bins, read_header
from mstd.data import V1Frame, V1Spec
from mstd.mt import encode_mt
from mstd.projection import load_projection, project_counts
import visualize_mt_gratings as view

STAGES = ('projected-hz', 'pooled-hz', 'normalized')
HEIGHT = 1230


def compute_stages(spec, counts, duration_ms, projection, radius):
    rates = project_counts(spec, counts, duration_ms, projection)
    n = spec.width*spec.height
    field = encode_mt(spec, V1Frame(0., rates), radius, return_channels=True)
    return (tuple(rates[d*n:(d+1)*n] for d in range(8)),
            field.pooled_rates_hz, field.direction_activity)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--stimulus', help='actual stimulus path; must match manifest SHA256')
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--spikes', help='V1 W x H x 84 SpikeMonitor DAT')
    source.add_argument('--cache', help='stages.json from an earlier completed comparison')
    p.add_argument('--projection-config', default='configs/v1_projection.json')
    p.add_argument('--projection', default='motionProj1')
    p.add_argument('--radius', type=int, default=1)
    p.add_argument('--output', required=True)
    p.add_argument('--max-frames', type=int)
    p.add_argument('--rate-max', type=float, default=300., help='shared fixed Hz scale for stages 1 and 2')
    p.add_argument('--activity-max', type=float, default=1.)
    p.add_argument('--colormap', choices=['hot', 'jet'], default='jet')
    p.add_argument('--playback-rate', type=float, default=1.)
    p.add_argument('--progress-every', type=int, default=20)
    p.add_argument('--ffmpeg', default='ffmpeg')
    a = p.parse_args(argv)
    if (a.radius < 0 or a.progress_every < 0 or a.max_frames is not None and a.max_frames <= 0
            or any(not math.isfinite(v) or v <= 0 for v in (a.rate_max, a.activity_max, a.playback_rate))):
        p.error('invalid frame, radius, progress, scale or playback argument')
    ffmpeg = shutil.which(a.ffmpeg)
    if not ffmpeg:
        p.error('ffmpeg is required')
    manifest = Path(a.manifest)
    meta = json.loads(manifest.read_text(encoding='utf-8-sig'))
    direction_schema = (meta.get('schema') == 'matlab-sequential-binary-gratings-v2'
                        or meta.get('schema') == 'visual-stimulus-sequence-v1' and meta.get('suite_type') == 'texture')
    if (not direction_schema
            or meta.get('coordinates') != 'x-right_y-down; clockwise from right'):
        raise ValueError('requires matching continuous MATLAB v2 manifest')
    run, ms = meta['runs'], meta['frame_ms']
    stimulus = Path(a.stimulus) if a.stimulus else manifest.parent/run['stimulus']
    info = inspect_stimulus(stimulus)
    if (info['width'], info['height'], info['frames'], info['channels']) != (meta['width'], meta['height'], run['frames'], 1):
        raise ValueError('stimulus dimensions/frames mismatch; use its own manifest')
    if ms <= 0 or run['duration_ms'] != run['frames']*ms or file_sha256(stimulus) != run['stimulus_sha256']:
        raise ValueError('stimulus checksum/timing mismatch; do not force an unrelated manifest')
    segments = view.load_segments(meta, manifest.parent)
    manifest_hash = file_sha256(manifest)
    segments_hash = file_sha256(manifest.parent/run['segments'])
    cached = json.loads(Path(a.cache).read_text(encoding='utf-8')) if a.cache else None
    projection = None
    if cached:
        if (cached['schema'] != 'mt-stage-cache-v1' or cached['manifest_sha256'] != manifest_hash
                or cached['segments_sha256'] != segments_hash or cached['stage_order'] != list(STAGES)):
            raise ValueError('cache does not match manifest/segments/stages')
        directions = tuple(cached['directions_rad'])
    else:
        projection = load_projection(a.projection_config, a.projection)
        directions = projection.directions_rad
    spec = V1Spec(meta['width'], meta['height'], directions, 1, 1, 0, 0)
    n = spec.width*spec.height
    order = [min(range(8), key=lambda j: abs((math.degrees(directions[j])-d*45+180)%360-180)) for d in range(8)]
    if len(set(order)) != 8 or any(abs((math.degrees(directions[j])-d*45+180)%360-180)>1e-5 for d,j in enumerate(order)):
        raise ValueError('directions must be the eight multiples of 45 degrees')
    total = min(run['frames'], a.max_frames or run['frames'])
    if cached:
        if (cached['width'], cached['height'], cached['frame_ms']) != (spec.width, spec.height, ms):
            raise ValueError('cache dimensions or timing mismatch')
        total = min(total, cached['frames'])
        cache_path = Path(a.cache).parent/cached['data_file']
        if cache_path.stat().st_size != cached['frames']*3*spec.frame_size*4 or file_sha256(cache_path) != cached['data_sha256']:
            raise ValueError('cache data size/hash mismatch')
    else:
        header = read_header(a.spikes)
        if (header.width, header.height, header.depth) != (spec.width, spec.height, 84):
            raise ValueError('V1 header must match stimulus W x H x 84')
    output = Path(a.output)
    output.mkdir(parents=True, exist_ok=False)
    command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', f'{view.W}x{HEIGHT}', '-r', str(1000/ms*a.playback_rate), '-i', '-', '-an',
               '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(output/'mt_stages.mp4')]
    bins = iter_bins(a.spikes, run['duration_ms'], ms) if a.spikes else None
    digest = hashlib.sha256()
    start, index, count = perf_counter(), 0, 0
    print(f'[start] {total} frames; one V1 pass, 3 stages, fixed scales Hz={a.rate_max}, activity={a.activity_max}', flush=True)
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        with ExitStack() as stack:
            source_image = stack.enter_context(stimulus.open('rb'))
            source_image.seek(25)
            cache_stream = stack.enter_context(cache_path.open('rb') if cached else (output/'stages.f32').open('xb'))
            table = stack.enter_context((output/'stage_statistics.csv').open('x', encoding='utf-8', newline=''))
            fields = ['frame', 'start_ms', 'end_ms', 'segment', 'is_blank', 'expected_deg', 'stage', 'units',
                      'winner_deg', 'spatial_min', 'spatial_max', 'std_at_expected', 'fraction_at_or_above_color_max']
            fields += [f'mean_{d}_deg' for d in range(0, 360, 45)]
            writer = csv.DictWriter(table, fieldnames=fields)
            writer.writeheader()
            for i in range(total):
                while i*ms >= segments[index]['end_ms']:
                    index += 1
                segment = segments[index]
                pixels = source_image.read(n)
                if len(pixels) != n:
                    raise ValueError('truncated stimulus frame')
                if cached:
                    values = array('f')
                    values.frombytes(cache_stream.read(spec.frame_size*3*4))
                    if sys.byteorder != 'little':
                        values.byteswap()
                    stages = tuple(tuple(values[(s*8+d)*n:(s*8+d+1)*n] for d in range(8)) for s in range(3))
                else:
                    t, width, counts = next(bins)
                    if t != i*ms or width != ms:
                        raise ValueError('spike/stimulus timing mismatch')
                    stages = compute_stages(spec, counts, width, projection, a.radius)
                    # Store and draw the same float32 samples; cache replay is pixel-identical.
                    values = array('f', (v for stage in stages for channel in stage for v in channel))
                    if sys.byteorder != 'little':
                        values.byteswap()
                    data = values.tobytes()
                    digest.update(data)
                    cache_stream.write(data)
                    if sys.byteorder != 'little':
                        values.byteswap()
                    stages = tuple(tuple(values[(s*8+d)*n:(s*8+d+1)*n] for d in range(8)) for s in range(3))
                moving = not segment['is_blank'] and segment['speed_px_per_frame'] > 0
                expected = segment['direction_slot_deg'] % 360 if moving else None
                parts = []
                for s, stage in enumerate(stages):
                    ordered = [stage[j] for j in order]
                    maximum = a.activity_max if s == 2 else a.rate_max
                    panel = view.render(pixels, ordered, spec, segment, i, total, ms, maximum, STAGES[s], a.colormap)
                    if not parts:
                        view.rect(panel, 290, 210, 1120, 24, (15, 21, 33))
                        view.text(panel, 295, 216, 'COMPARE: PROJECTED HZ / POOLED HZ / NORMALIZED ACTIVITY')
                        parts.append(panel[:315*view.W*3])
                    parts.append(panel[315*view.W*3:620*view.W*3])
                    means = [sum(ch)/n for ch in ordered]
                    expected_map = ordered[int(expected//45)] if expected is not None else None
                    expected_mean = sum(expected_map)/n if expected_map is not None else None
                    flat = [v for ch in ordered for v in ch]
                    row = dict(frame=i+1, start_ms=i*ms, end_ms=(i+1)*ms, segment=segment['segment_index'],
                        is_blank=segment['is_blank'], expected_deg=expected, stage=STAGES[s],
                        units='normalized' if s == 2 else 'Hz',
                        winner_deg=45*max(range(8), key=means.__getitem__) if max(means)>1e-8 else None,
                        spatial_min=min(flat), spatial_max=max(flat),
                        std_at_expected=math.sqrt(sum((v-expected_mean)**2 for v in expected_map)/n) if expected_map is not None else None,
                        fraction_at_or_above_color_max=sum(v>=maximum for v in flat)/len(flat))
                    row.update({f'mean_{d*45}_deg': means[d] for d in range(8)})
                    writer.writerow(row)
                canvas = b''.join(parts)
                process.stdin.write(canvas)
                offset = min(1000, ((segment['end_ms']-segment['start_ms'])//(2*ms))*ms)
                if i == 0 or i*ms == segment['start_ms']+offset:
                    view.png(output/f'stage_segment_{segment["segment_index"]:03d}.png', canvas, view.W, HEIGHT)
                count += 1
                if a.progress_every and (count == 1 or count % a.progress_every == 0 or count == total):
                    table.flush()
                    cache_stream.flush()
                    elapsed = perf_counter()-start
                    print(f'[progress] {count}/{total}, elapsed={elapsed:.1f}s, remaining~{elapsed/count*(total-count):.1f}s', flush=True)
            if bins is not None and total == run['frames'] and next(bins, None) is not None:
                raise ValueError('unexpected extra bin')
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError('ffmpeg failed')
    except BaseException:
        if process.poll() is None:
            process.kill()
        process.wait()
        raise
    finally:
        if bins is not None:
            bins.close()
    if not cached:
        result = dict(schema='mt-stage-cache-v1', stage_order=list(STAGES), width=spec.width, height=spec.height,
            frames=count, frame_ms=ms, directions_rad=directions, data_file='stages.f32',
            data_sha256=digest.hexdigest(), encoding='little-endian float32 T,stage,D,Y,X',
            manifest_sha256=manifest_hash, segments_sha256=segments_hash,
            stimulus_sha256=run['stimulus_sha256'], projection_config_sha256=file_sha256(a.projection_config),
            projection_name=a.projection, pool_radius=a.radius, spike_source=str(Path(a.spikes).resolve()),
            spike_bytes=Path(a.spikes).stat().st_size, spike_events=header.event_count,
            full_event_validation=count == run['frames'], units=['projected Hz', 'pooled input Hz', 'normalized activity'])
        (output/'stages.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    config = dict(frames=count, complete_manifest=count == run['frames'], stimulus=str(stimulus.resolve()),
        manifest_sha256=manifest_hash, source_cache=str(Path(a.cache).resolve()) if cached else 'stages.json',
        rate_max=a.rate_max, activity_max=a.activity_max, colormap=a.colormap, playback_rate=a.playback_rate,
        stage_order=list(STAGES), output_direction_order_deg=list(range(0,360,45)),
        response_available='window end', statistics_roi='full image; includes edge pixels',
        limitation='Rates in first two rows are not measured Python MT spikes. Spatial std is not an accuracy score.')
    (output/'video_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(f'[done] {output}/mt_stages.mp4', flush=True)


if __name__ == '__main__':
    main()
