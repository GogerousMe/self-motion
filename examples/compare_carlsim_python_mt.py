"""Compare actual CARLsim MT spikes with saved Python Stage 2 monitors.

NumPy + Pillow + ffmpeg. No torch/GPU needed for this offline renderer.
Different grid sizes are displayed at native aspect ratio, never spatially compared.
CARLsim direction order, cadence, start alignment and stimulus pairing must be audited.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mstd.calibration import file_sha256, inspect_stimulus
from mstd.carlsim import read_header
from mstd.image_sequence import IMAGE_SUFFIXES, natural_key

ANGLES = np.arange(8)*45
NAMES = ['RIGHT', 'RIGHT-DOWN', 'DOWN', 'LEFT-DOWN', 'LEFT', 'LEFT-UP', 'UP', 'RIGHT-UP']


def direction_order(degrees):
    degrees = np.asarray(degrees, dtype=float) % 360
    if degrees.shape != (8,) or not np.isfinite(degrees).all():
        raise ValueError('eight finite direction angles required')
    distance = abs((ANGLES[:, None]-degrees[None, :]+180) % 360-180)
    order = distance.argmin(axis=1)
    if len(set(order)) != 8 or np.max(distance[np.arange(8), order]) > 1e-4:
        raise ValueError('directions must uniquely cover 0:45:315 degrees')
    return order


def bin_spikes(path, frame_ms, duration_ms=None):
    header = read_header(path)
    if header.depth != 8:
        raise ValueError('actual MT DAT must have depth=8, not V1 depth=84')
    events = np.memmap(path, dtype=header.endian+'i4', mode='r', offset=20,
                       shape=(header.event_count, 2)) if header.event_count else np.empty((0, 2), int)
    times, ids = events[:, 0], events[:, 1]
    if (np.any(times < 0) or np.any(np.diff(times.astype(np.int64)) < 0)
            or np.any(ids < 0) or np.any(ids >= header.neurons)):
        raise ValueError('invalid spike event time/order/id')
    last = int(times[-1]) if len(times) else None
    inferred = duration_ms is None
    if inferred:
        if last is None:
            raise ValueError('silent DAT requires explicit --carlsim-duration-ms')
        # Event-supported extent only: cannot infer a silent simulation tail.
        duration_ms = (last//frame_ms+1)*frame_ms
    if duration_ms <= 0 or duration_ms % frame_ms or (last is not None and last >= duration_ms):
        raise ValueError('duration must cover every event and align with frame_ms')
    count = duration_ms//frame_ms
    rates = np.empty((count, 8, header.height, header.width), np.float32)
    boundaries = np.searchsorted(times, np.arange(count+1, dtype=np.int64)*frame_ms)
    for i in range(count):
        counts = np.bincount(ids[boundaries[i]:boundaries[i+1]], minlength=header.neurons)
        rates[i] = counts.reshape(8, header.height, header.width)*(1000/frame_ms)
    return rates, dict(width=header.width, height=header.height, events=header.event_count,
                      last_event_ms=last, duration_ms=duration_ms, duration_inferred=inferred)


def load_python(root):
    report = json.loads((root/'model_report.json').read_text(encoding='utf-8'))
    meta_path = root/'monitors/mt_pooling/metadata.json'
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    model = report['model']
    shape = (report['frames'], 8, model['height'], model['width'])
    dt = model['frame_interval_s']
    path = meta_path.parent/'values.f32'
    if (report.get('complete') is not True or meta.get('complete') is not True
            or meta.get('schema') != 'mstd-layer-monitor-v1'
            or meta.get('layout') != 'TDYX' or meta.get('dtype') != 'float32-le'
            or meta.get('units') != 'Hz' or meta.get('file') != 'values.f32'
            or meta.get('coordinates') != 'x-right_y-down'
            or tuple(meta['shape']) != shape or path.stat().st_size != math.prod(shape)*4
            or meta['sample_indices'] != list(range(shape[0]))
            or len(meta['timestamps_s']) != shape[0]
            or not np.allclose(meta['timestamps_s'], np.arange(shape[0])*dt, atol=1e-9, rtol=0)):
        raise ValueError('incomplete, sparse, malformed or misaligned Stage 2 monitor')
    values = np.memmap(path, dtype='<f4', mode='r', shape=shape)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('invalid Python rates')
    order = direction_order(np.degrees(meta['directions_rad']))
    return values[:, order], report


def profile(values):
    total = float(np.sum(values))
    if total <= 1e-12:
        return None
    p = np.asarray(values)/total
    z = np.sum(p*np.exp(1j*np.radians(ANGLES)))
    return dict(winner_deg=int(ANGLES[np.argmax(p)]), fractions=p.tolist(),
                resultant=float(abs(z)), vector_deg=float(np.degrees(np.angle(z)) % 360) if abs(z)>1e-8 else None)


def font(size):
    for path in ('C:/Windows/Fonts/arial.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def jet(values, maximum):
    v = np.clip(values/maximum, 0, 1)
    return np.rint(np.clip(1.5-abs(4*v[..., None]-np.array([3, 2, 1])), 0, 1)*255).astype('uint8')


def paste_fit(canvas, image, box):
    factor = min(box[2]/image.width, box[3]/image.height)
    image = image.resize((max(1, round(image.width*factor)), max(1, round(image.height*factor))),
                         Image.Resampling.NEAREST)
    canvas.paste(image, (box[0]+(box[2]-image.width)//2, box[1]+(box[3]-image.height)//2))


def render(source, maps, means, maxima, index, total, frame_ms, border, mismatch, inferred):
    canvas = Image.new('RGB', (1440, 940), (15, 21, 33))
    draw = ImageDraw.Draw(canvas)
    def text(x, y, s, size=19, color='white'):
        draw.text((x, y), s, font=font(size), fill=color)
    text(20, 12, 'INPUT / CARLSIM MT SPIKES / PYTHON MT STAGE 2', 28)
    paste_fit(canvas, source, (20, 53, 260, 208))
    text(310, 62, f'FRAME {index+1}/{total}   WINDOW [{index*frame_ms/1000:.2f}, {(index+1)*frame_ms/1000:.2f}) s', 24)
    text(310, 105, 'Direction columns: clockwise from right. Cyan: strongest ROI mean.')
    text(310, 138, f'ROI excludes {border} native pixels per edge. No temporal shift or spatial smoothing.')
    text(310, 171, 'CARLsim pairing / 50 ms-style cadence / channel labels: assumed from supplied context.', 18, '#ffcc55')
    text(310, 202, 'DIFFERENT INPUT GRIDS: exploratory preference comparison.' if mismatch else 'Matching grids do not prove matching stimulus preprocessing.', 18, '#ffcc55')
    text(310, 233, 'Duration inferred from last spike; silent tail unknown.' if inferred else 'CARLsim duration explicitly supplied.', 18, '#ffcc55')
    for row, (values, avg, maximum) in enumerate(zip(maps, means, maxima)):
        top = 290+row*315
        title = 'CARLSIM MT: OBSERVED SPIKE RATE' if row == 0 else 'PYTHON MT: STAGE 2 POOLED DIRECTION RATE (NOT MT SPIKES)'
        text(20, top, f'{title}   {values.shape[2]}x{values.shape[1]}', 22)
        winner = int(np.argmax(avg)) if max(avg)>1e-12 else None
        for d in range(8):
            x = 20+d*177
            text(x, top+34, f'{d*45} DEG {NAMES[d]}', 13)
            paste_fit(canvas, Image.fromarray(jet(values[d], maximum)), (x, top+60, 164, 168))
            text(x, top+237, f'MEAN {avg[d]:.2f} Hz', 17)
            if winner == d:
                draw.line((x, top+231, x+164, top+231), fill='#55e8ff', width=4)
        bar = Image.fromarray(jet(np.linspace(0, maximum, 260)[None, :], maximum)).resize((260, 18))
        canvas.paste(bar, (285, top+274))
        text(20, top+270, 'FIXED SCALE: 0', 18)
        text(560, top+270, f'{maximum:.2f} Hz (clipped for display only; fixed over all frames)', 18)
    text(20, 920, 'Rows use independent fixed scales; compare numerical profiles, not absolute color brightness.', 15, '#ffcc55')
    return canvas


def chart(path, car, py):
    image = Image.new('RGB', (1200, 600), 'white')
    draw = ImageDraw.Draw(image)
    draw.text((30, 15), 'Direction fractions after warmup (no ground-truth direction labels)', font=font(25), fill='black')
    colors = ('#e17823', '#237abf')
    profiles = [profile(car), profile(py)]
    for d in range(8):
        x = 85+d*137
        draw.text((x-10, 505), f'{d*45} deg', font=font(20), fill='black')
        for j, p in enumerate(profiles):
            value = p['fractions'][d] if p else 0
            h = value*390
            draw.rectangle((x+j*45, 490-h, x+j*45+37, 490), fill=colors[j])
            draw.text((x+j*45, 465-h), f'{value:.2f}', font=font(15), fill=colors[j])
    draw.text((40, 550), 'Orange: CARLsim spike rate fraction    Blue: Python Stage 2 fraction', font=font(23), fill='black')
    draw.text((40, 580), 'Different grids / unverified CARLsim input pairing: exploratory comparison only.', font=font(16), fill='black')
    image.save(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spikes', type=Path, required=True)
    p.add_argument('--recording', type=Path, required=True)
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--stimulus', type=Path, required=True, help='exact DAT used by Python, SHA256 checked')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--frame-ms', type=int, default=50, help='CARLsim cadence assumption; must match Python')
    p.add_argument('--carlsim-duration-ms', type=int, help='actual duration; otherwise event-supported extent is inferred')
    p.add_argument('--carlsim-directions', type=float, nargs=8, default=[0,315,270,225,180,135,90,45])
    p.add_argument('--warmup-ms', type=int, default=1000)
    p.add_argument('--border', type=int, default=4)
    p.add_argument('--rate-max', type=float, nargs=2, help='fixed CARLsim / Python color maxima; default separate 99.5th percentiles')
    args = p.parse_args()
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('ffmpeg must be on PATH')
    if args.frame_ms <= 0 or args.warmup_ms < 0 or args.warmup_ms % args.frame_ms or args.border < 0:
        raise ValueError('invalid frame/warmup/border settings')
    py, report = load_python(args.recording)
    info = inspect_stimulus(args.stimulus)
    if (file_sha256(args.stimulus) != report['input']['stimulus_sha256']
            or (info['width'], info['height']) != (py.shape[3], py.shape[2]) or info['channels'] != 1
            or info['frames'] < len(py)
            or not math.isclose(report['model']['frame_interval_s']*1000, args.frame_ms, abs_tol=1e-8)):
        raise ValueError('Python stimulus hash, grid, channel count or timing mismatch')
    images = sorted((f for f in args.images.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES), key=lambda f:(natural_key(f.name),f.name))
    if len(images) < len(py):
        raise ValueError('not enough source images')
    raw = np.memmap(args.stimulus, dtype='u1', mode='r', offset=25,
                    shape=(info['frames'], info['height'], info['width']))
    # Verify every displayed source frame against the actual Python stimulus.
    for i, image in enumerate(images[:len(py)]):
        with Image.open(image) as source:
            if not np.array_equal(np.asarray(source.convert('L')), raw[i]):
                raise ValueError(f'image {i+1} does not match Python DAT grayscale pixels')
    car, acquisition = bin_spikes(args.spikes, args.frame_ms, args.carlsim_duration_ms)
    car = car[:, direction_order(args.carlsim_directions)]
    total = min(len(car), len(py))
    skip = args.warmup_ms//args.frame_ms
    if skip >= total or args.border*2 >= min(*car.shape[2:], *py.shape[2:]):
        raise ValueError('warmup/border leaves no evaluation data')
    car, py = car[:total], py[:total]
    cut = slice(args.border, -args.border) if args.border else slice(None)
    means = [v[:, :, cut, cut].mean(axis=(2,3), dtype=np.float64) for v in (car,py)]
    maxima = args.rate_max or [max(1., float(np.percentile(v, 99.5))) for v in (car,py)]
    if not all(math.isfinite(v) and v>0 for v in maxima):
        raise ValueError('positive finite color maxima required')
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for i, (a,b) in enumerate(zip(*means)):
        pa,pb = profile(a),profile(b)
        row = dict(frame=i+1, start_ms=i*args.frame_ms, end_ms=(i+1)*args.frame_ms,
                   evaluation=i>=skip, carlsim_winner_deg=pa['winner_deg'] if pa else None,
                   python_winner_deg=pb['winner_deg'] if pb else None,
                   total_variation=float(np.abs(np.array(pa['fractions'])-pb['fractions']).sum()/2) if pa and pb else None)
        for label, vector in (('carlsim',a),('python',b)):
            row.update({f'{label}_{d*45}_mean_hz':float(v) for d,v in enumerate(vector)})
        rows.append(row)
    with (args.output/'direction_statistics.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    aggregate = [profile(v[skip:].mean(axis=0)) for v in means]
    valid = [r for r in rows[skip:] if r['total_variation'] is not None]
    summary = dict(schema='carlsim-python-mt-comparison-v1', frames=total, frame_ms=args.frame_ms,
                   python_frames=report['frames'], carlsim=acquisition, warmup_ms=args.warmup_ms,
                   border_native_pixels=args.border, python_stage='mt_pooling', direction_order_deg=ANGLES.tolist(),
                   carlsim_native_direction_deg=args.carlsim_directions, source_images_match_python_dat=True,
                   input_sha256=file_sha256(args.stimulus), spikes_sha256=file_sha256(args.spikes),
                   python_monitor_sha256=file_sha256(args.recording/'monitors/mt_pooling/values.f32'),
                   carlsim_grid=list(car.shape[2:]), python_grid=list(py.shape[2:]), color_max_hz=maxima,
                   aggregate_carlsim=aggregate[0], aggregate_python=aggregate[1],
                   active_comparison_frames=len(valid),
                   winner_agreement=float(np.mean([r['carlsim_winner_deg']==r['python_winner_deg'] for r in valid])) if valid else None,
                   mean_total_variation=float(np.mean([r['total_variation'] for r in valid])) if valid else None,
                   limitations=['CARLsim stimulus identity/preprocessing, simulation start and cadence not independently verified.',
                                'CARLsim labels assume local source comments: right/up-right/up/...; not empirically calibrated.',
                                'Different resolutions change pixel speeds and receptive fields; no pixelwise accuracy claimed.',
                                'No ground-truth motion direction for natural-scene images; agreement is not accuracy.',
                                'No temporal shift, smoothing, or neural simulation performed.',
                                'Last spike is not simulation duration; inferred extent may omit silent tail.',
                                'Python Stage 2 is pooled continuous readout, not observed MT firing rate.'])
    chart(args.output/'direction_profiles.png', *(v[skip:].mean(axis=0) for v in means))
    command = [ffmpeg,'-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s','1440x940',
               '-r',str(1000/args.frame_ms),'-i','pipe:0','-an','-c:v','libx264','-crf','18',
               '-pix_fmt','yuv420p','-movflags','+faststart',str(args.output/'mt_comparison.mp4')]
    with (args.output/'ffmpeg.log').open('w') as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        try:
            for i in range(total):
                with Image.open(images[i]) as image:
                    canvas = render(image.convert('RGB'), (car[i],py[i]), [v[i] for v in means], maxima,
                                    i,total,args.frame_ms,args.border,car.shape[2:]!=py.shape[2:],acquisition['duration_inferred'])
                if i in (0,total-1, min(total-1,skip)):
                    canvas.save(args.output/f'frame_{i+1:04d}.png')
                process.stdin.write(canvas.tobytes())
                if i%20 == 0: print(f'[comparison] {i+1}/{total}',flush=True)
        finally:
            process.stdin.close()
            code = process.wait()
        if code:
            raise RuntimeError(f'ffmpeg failed ({code}); see ffmpeg.log')
    summary['complete'] = True
    (args.output/'comparison_report.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
