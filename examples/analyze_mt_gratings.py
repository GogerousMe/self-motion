"""Analyze continuous MATLAB v2 gratings through the actual Python MT encoder.

Run from code/: PYTHONPATH=src python examples/analyze_mt_gratings.py --help
Only standard-library modules are required. No MSTd or camera intrinsics used.
"""
import argparse
import csv
import json
import sys
from time import perf_counter
from math import atan2, degrees, hypot, isfinite
from pathlib import Path

from mstd.calibration import file_sha256, inspect_stimulus
from mstd.carlsim import iter_bins, read_header
from mstd.data import V1Frame, V1Spec
from mstd.mt import encode_mt
from mstd.projection import load_projection, project_counts


def angle(x, y):
    return degrees(atan2(y, x)) % 360 if hypot(x, y) > 1e-12 else None


def error(actual, expected):
    return abs((actual - expected + 180) % 360 - 180) if actual is not None and expected is not None else None


def analyze(manifest, spikes, output, projection_config, projection='motionProj1',
            warmup_ms=1000, border=4, radius=1, tolerance=22.5,
            min_evidence=.02, min_coherence=.1, progress_every=100):
    if type(progress_every) is not int or progress_every < 0:
        raise ValueError('progress_every must be a nonnegative integer')
    started = perf_counter()

    def log(message):
        if progress_every:
            print(message, file=sys.stderr, flush=True)

    log('[preflight] Reading manifest and checking stimulus checksum / spike header...')
    manifest, spikes, output = Path(manifest), Path(spikes), Path(output)
    meta = json.loads(manifest.read_text(encoding='utf-8-sig'))
    if not (meta.get('schema') == 'matlab-sequential-binary-gratings-v2'
            or meta.get('schema') == 'visual-stimulus-sequence-v1' and meta.get('suite_type') == 'texture'):
        raise ValueError('requires matlab-sequential-binary-gratings-v2 manifest')
    if meta.get('coordinates') != 'x-right_y-down; clockwise from right':
        raise ValueError('unsupported image coordinates')
    run = meta['runs']
    frame_ms, duration = meta['frame_ms'], run['duration_ms']
    w, h = meta['width'], meta['height']
    if any(type(v) is not int for v in (w, h, frame_ms, duration, warmup_ms, border, radius)):
        raise ValueError('integer dimensions, timing, border and radius required')
    if min(w, h, frame_ms, duration) <= 0 or duration != run['frames'] * frame_ms:
        raise ValueError('invalid dimensions or duration')
    if warmup_ms < 0 or warmup_ms % frame_ms or radius < 0 or not radius <= border < min(w, h)/2:
        raise ValueError('warmup must align to frames; radius <= border < half grid size required')
    if not all(isfinite(v) for v in (tolerance, min_evidence, min_coherence)) or not (
            0 <= tolerance <= 180 and 0 <= min_evidence <= 1 and 0 <= min_coherence <= 1):
        raise ValueError('invalid thresholds')
    stimulus = manifest.parent / run['stimulus']
    info = inspect_stimulus(stimulus)
    if (info['width'], info['height'], info['frames'], info['channels']) != (w, h, run['frames'], 1):
        raise ValueError('stimulus dimensions mismatch')
    if file_sha256(stimulus) != run['stimulus_sha256']:
        raise ValueError('stimulus checksum mismatch')
    header = read_header(spikes)
    if (header.width, header.height, header.depth) != (w, h, 84):
        raise ValueError('expected matching W x H x 84 V1 spike grid')
    segments_path = manifest.parent / run['segments']
    with segments_path.open(encoding='utf-8-sig', newline='') as stream:
        segments = list(csv.DictReader(stream))
    previous_end = 0
    for row in segments:
        for key in ('segment_index', 'first_frame', 'last_frame', 'start_ms', 'end_ms', 'is_blank'):
            row[key] = int(row[key])
        for key in ('direction_slot_deg', 'speed_px_per_frame'):
            row[key] = float(row[key])
        if not all(isfinite(row[k]) for k in ('direction_slot_deg', 'speed_px_per_frame')):
            raise ValueError('nonfinite segment label')
        if row['is_blank'] not in (0, 1) or row['speed_px_per_frame'] < 0:
            raise ValueError('invalid blank or speed label')
        if (row['start_ms'] != previous_end or row['end_ms'] <= previous_end
                or row['start_ms'] != (row['first_frame']-1)*frame_ms
                or row['end_ms'] != row['last_frame']*frame_ms):
            raise ValueError('segments must cover contiguous frame-aligned intervals')
        previous_end = row['end_ms']
    if not segments or previous_end != duration:
        raise ValueError('segments must cover entire recording')
    proj = load_projection(projection_config, projection)
    spec = V1Spec(w, h, proj.directions_rad, 1, 1, 0, 0)  # No geometric heading inference.
    roi = [y*w+x for y in range(border, h-border) for x in range(border, w-border)]
    labels = [degrees(a) % 360 for a in proj.directions_rad]
    accum = [dict(bins=0, x=0., y=0., evidence=0., rates=[0.]*8) for _ in segments]
    fields = ['start_ms', 'end_ms', 'segment_index', 'is_blank', 'speed_px_per_frame',
              'expected_direction_deg', 'evaluation_bin', 'mt_direction_deg', 'mt_error_deg',
              'mt_evidence', 'mt_coherence'] + [f'projected_ch{i}_hz' for i in range(8)]
    output.mkdir(parents=True, exist_ok=False)
    index = 0
    processed = 0
    processing_started = perf_counter()
    log(f'[start] {w}x{h}x84, {run["frames"]} bins, {header.event_count} events; '
        f'simulated duration={duration/1000:g}s. Output: {output}')
    # A failed run leaves its partial CSV for diagnosis, but no successful report.json.
    with (output/'frames.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        stream.flush()
        for start, width, counts in iter_bins(spikes, duration, frame_ms):
            while start >= segments[index]['end_ms']:
                index += 1
            seg = segments[index]
            moving = not seg['is_blank'] and seg['speed_px_per_frame'] > 0
            expected = seg['direction_slot_deg'] % 360 if moving else None
            rates = project_counts(spec, counts, width, proj)
            mt = encode_mt(spec, V1Frame(start/1000, rates), radius)
            x = sum(mt.vx[p] for p in roi)/len(roi)
            y = sum(mt.vy[p] for p in roi)/len(roi)
            evidence = sum(hypot(mt.vx[p], mt.vy[p]) for p in roi)/len(roi)
            coherence = min(1., hypot(x, y)/evidence) if evidence else 0.
            direction = angle(x, y)
            evaluation = start >= seg['start_ms'] + warmup_ms
            means = [sum(rates[d*w*h+p] for p in roi)/len(roi) for d in range(8)]
            row = dict(start_ms=start, end_ms=start+width, segment_index=seg['segment_index'],
                       is_blank=seg['is_blank'], speed_px_per_frame=seg['speed_px_per_frame'],
                       expected_direction_deg=expected, evaluation_bin=evaluation,
                       mt_direction_deg=direction, mt_error_deg=error(direction, expected),
                       mt_evidence=evidence, mt_coherence=coherence)
            row.update({f'projected_ch{i}_hz': value for i, value in enumerate(means)})
            writer.writerow(row)
            processed += 1
            if progress_every and (processed == 1 or processed % progress_every == 0 or processed == run['frames']):
                stream.flush()
                elapsed = perf_counter()-processing_started
                fps = processed/elapsed if elapsed > 0 else 0.
                eta = (run['frames']-processed)/fps if fps else 0.
                log(f'[progress] {processed}/{run["frames"]} bins '
                    f'({100*processed/run["frames"]:.1f}%), segment={seg["segment_index"]}, '
                    f'simulated={(start+width)/1000:g}s, elapsed={elapsed:.1f}s, '
                    f'{fps:.2f} bins/s, remaining~{eta:.1f}s')
            if evaluation:
                a = accum[index]
                a['bins'] += 1
                a['x'] += x
                a['y'] += y
                a['evidence'] += evidence
                a['rates'] = [old+new for old, new in zip(a['rates'], means)]
    summaries = []
    for seg, a in zip(segments, accum):
        moving = not seg['is_blank'] and seg['speed_px_per_frame'] > 0
        expected = seg['direction_slot_deg'] % 360 if moving else None
        n = a['bins']
        direction = angle(a['x'], a['y']) if n else None
        evidence = a['evidence']/n if n else 0.
        coherence = min(1., hypot(a['x'], a['y'])/a['evidence']) if a['evidence'] else 0.
        responsive = evidence >= min_evidence and coherence >= min_coherence and direction is not None
        rates = [v/n for v in a['rates']] if n else [0.]*8
        winner = labels[max(range(8), key=rates.__getitem__)] if sum(rates) else None
        err = error(direction, expected)
        summaries.append(dict(**seg, evaluation_bins=n, expected_direction_deg=expected,
            mt_direction_deg=direction, mt_error_deg=err, mt_evidence=evidence,
            mt_coherence=coherence, responsive=responsive,
            within_tolerance=(responsive and err <= tolerance) if moving else None,
            projected_winner_direction_deg=winner, projected_direction_mean_hz=rates))
    moving_rows = [s for s in summaries if s['expected_direction_deg'] is not None]
    speed_results = []
    for speed in sorted({s['speed_px_per_frame'] for s in moving_rows}):
        group = [s for s in moving_rows if s['speed_px_per_frame'] == speed]
        complete = {s['expected_direction_deg'] for s in group} == set(range(0, 360, 45))
        speed_results.append(dict(speed_px_per_frame=speed, eight_directions_present=complete,
                                  passed=complete and all(s['within_tolerance'] for s in group)))
    log('[finalize] Frames complete. Hashing source spike file and writing report...')
    report = dict(schema='continuous-grating-mt-report-v1',
        manifest_sha256=file_sha256(manifest), segments_sha256=file_sha256(segments_path),
        spikes_sha256=file_sha256(spikes), projection_config_sha256=file_sha256(projection_config),
        projection_name=projection, direction_channel_angles_deg=labels,
        frame_ms=frame_ms, duration_ms=duration, warmup_ms_per_segment=warmup_ms,
        border=border, pool_radius=radius, tolerance_deg=tolerance,
        min_evidence=min_evidence, min_coherence=min_coherence,
        speed_results=speed_results, moving_directions_passed=bool(speed_results) and all(s['passed'] for s in speed_results),
        segments=summaries, limitations=[
            'Projected Hz channels are before MT; MT metrics use actual encode_mt pooling and normalization.',
            'Engineering direction screen only; inspect blank/static responses and transient frames separately.',
            'Continuous conditions have history effects. No MSTd heading or self-motion accuracy claim.',
            'Stimulus/response pairing and simulation timing must be confirmed from acquisition logs.'])
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    log(f'[done] {processed} bins, total elapsed={perf_counter()-started:.1f}s; {output}/report.json')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--spikes', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--projection-config', default='configs/v1_projection.json')
    parser.add_argument('--projection', default='motionProj1')
    parser.add_argument('--warmup-ms', type=int, default=1000)
    parser.add_argument('--border', type=int, default=4)
    parser.add_argument('--radius', type=int, default=1)
    parser.add_argument('--tolerance', type=float, default=22.5)
    parser.add_argument('--min-evidence', type=float, default=.02)
    parser.add_argument('--min-coherence', type=float, default=.1)
    parser.add_argument('--progress-every', type=int, default=100,
                        help='print progress to stderr every N bins (default 100; 0 disables)')
    args = vars(parser.parse_args())
    report = analyze(**args)
    print(json.dumps({k: report[k] for k in ('moving_directions_passed', 'speed_results')}))


if __name__ == '__main__':
    main()
