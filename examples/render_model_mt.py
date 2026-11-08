"""Render saved Python MT monitors, aligned to the exact input DAT.

Called by compare_mt_stages.py --recording. NumPy + ffmpeg; no torch,
matplotlib or CARLsim dependency. Memory maps avoid loading full recordings.
"""
import csv
import json
import math
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np

from mstd.calibration import file_sha256, inspect_stimulus
from mstd.project_paths import resolve_stimulus
import visualize_mt_gratings as view

STAGES = ('mt_projection', 'mt_pooling', 'mt_normalization')
UNITS = ('Hz', 'Hz', 'dimensionless')
WIDTH, HEIGHT = 1440, 1230


def load_recording(recording, stimulus=None, manifest=None, max_frames=None):
    """Validate pairing, cadence and channel order before creating any output."""
    root = Path(recording)
    report = json.loads((root/'model_report.json').read_text(encoding='utf-8'))
    if report.get('schema') != 'mstd-simulation-v1' or report.get('complete') is not True:
        raise ValueError('a completed run_model.py recording is required')
    model = report['model']
    dt = model['frame_interval_s']
    count = report['frames']
    if (type(count) is not int or count <= 0 or isinstance(dt, bool)
            or not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0):
        raise ValueError('invalid recording frame count or timing')
    if report['monitors']['every_n_frames'] != 1:
        raise ValueError('video requires every_n_frames=1; rerun the model to save every input frame')
    if stimulus is None:
        if report['input'].get('input_kind') != 'stimulus_dat':
            raise ValueError('--stimulus is required for a V1 replay recording')
        stimulus = report['input']['input_path']
    stimulus = resolve_stimulus(stimulus, report['input'].get('stimulus_sha256'))
    info = inspect_stimulus(stimulus)
    stimulus_hash = file_sha256(stimulus)
    if stimulus_hash != report['input'].get('stimulus_sha256'):
        raise ValueError('stimulus SHA256 differs from the input used by the model')
    if (model['width'], model['height']) != (info['width'], info['height']) or count > info['frames']:
        raise ValueError('model/stimulus dimensions or frame count disagree')
    if max_frames is not None and (type(max_frames) is not int or max_frames <= 0):
        raise ValueError('positive integer max_frames required')
    if max_frames is not None and max_frames > count:
        raise ValueError(f'only {count} model frames were recorded, but {max_frames} requested; '
                         'rerun run_model.py for the required duration before rendering')
    total = max_frames if max_frames is not None else count
    definitions = []
    directions = None
    for name, units in zip(STAGES, UNITS):
        metadata = root/'monitors'/name/'metadata.json'
        if not metadata.is_file():
            raise ValueError(f'missing {name} monitor; enable all three MT monitors and rerun the model')
        meta = json.loads(metadata.read_text(encoding='utf-8'))
        shape = [count, 8, info['height'], info['width']]
        if (meta.get('schema') != 'mstd-layer-monitor-v1' or meta.get('complete') is not True
                or meta.get('dtype') != 'float32-le' or meta.get('layout') != 'TDYX'
                or meta.get('shape') != shape or meta.get('units') != units
                or meta.get('file') != 'values.f32' or meta.get('coordinates') != 'x-right_y-down'):
            raise ValueError(f'invalid {name} monitor shape, layout or units')
        indices = meta.get('sample_indices', [])
        times = np.asarray(meta.get('timestamps_s', []), dtype=np.float64)
        if (len(indices) != count or any(type(v) is not int or v != i for i, v in enumerate(indices))
                or times.shape != (count,) or not np.isfinite(times).all()
                or not np.allclose(times, np.arange(count)*dt, atol=1e-9, rtol=0)):
            raise ValueError(f'{name} has missing or mismatched frame indices/timestamps')
        angles = np.asarray(meta.get('directions_rad', []), dtype=np.float64)
        if angles.shape != (8,) or not np.isfinite(angles).all():
            raise ValueError('eight finite direction angles required')
        if directions is not None and not np.array_equal(directions, angles):
            raise ValueError('MT monitor direction order differs across stages')
        directions = angles
        binary = metadata.parent/'values.f32'
        if binary.stat().st_size != math.prod(shape)*4:
            raise ValueError(f'{name} payload size mismatch')
        definitions.append((binary, shape, file_sha256(metadata)))
    degrees = np.degrees(directions) % 360
    errors = np.abs((degrees[None, :]-np.arange(8)[:, None]*45+180) % 360-180)
    order = errors.argmin(axis=1)
    if len(set(order.tolist())) != 8 or np.any(errors[np.arange(8), order] > 1e-5):
        raise ValueError('monitor directions must be the eight multiples of 45 degrees')
    segments = None
    manifest_hash = segments_hash = None
    if manifest:
        manifest = Path(manifest)
        meta = json.loads(manifest.read_text(encoding='utf-8-sig'))
        valid_schema = (meta.get('schema') == 'matlab-sequential-binary-gratings-v2'
                        or meta.get('schema') == 'visual-stimulus-sequence-v1' and meta.get('suite_type') == 'texture')
        if not valid_schema or meta.get('coordinates') != 'x-right_y-down; clockwise from right':
            raise ValueError('unsupported sequence manifest; omit --manifest to render without direction labels')
        run = meta['runs']
        if ((meta['width'], meta['height'], run['frames']) != (info['width'], info['height'], info['frames'])
                or run['stimulus_sha256'] != stimulus_hash
                or not math.isclose(meta['frame_ms'], dt*1000, abs_tol=1e-8, rel_tol=0)
                or run['duration_ms'] != run['frames']*meta['frame_ms']):
            raise ValueError('manifest/stimulus/model hash, size or timing mismatch')
        segments = view.load_segments(meta, manifest.parent)
        manifest_hash = file_sha256(manifest)
        segments_hash = file_sha256(manifest.parent/run['segments'])
    return dict(root=root, report=report, stimulus=stimulus, info=info, frames=total,
                dt=dt, order=order, definitions=definitions, segments=segments,
                stimulus_sha256=stimulus_hash, manifest_sha256=manifest_hash,
                segments_sha256=segments_hash)


class Renderer:
    """Fixed scales and nearest-neighbor maps, matching the old three-row layout."""
    def __init__(self, width, height, rate_max, activity_max, palette,titles=None,subtitle=None):
        self.rate_max, self.activity_max, self.palette = rate_max, activity_max, palette
        self.width, self.height = width, height
        self.glyphs = {}
        self.base = np.empty((HEIGHT, WIDTH, 3), np.uint8)
        self.base[:] = (15, 21, 33)
        self.text(self.base, 20, 14, 'PYTHON MODEL: INPUT STIMULUS AND MT RESPONSES', 3)
        self.text(self.base, 295, 216, subtitle or 'SAVED MODEL MONITORS - CONTINUOUS RESPONSES')
        self.text(self.base, 295, 246, 'YELLOW: EXPECTED   CYAN: STRONGEST CHANNEL MEAN', color=(140, 215, 235))
        titles = titles or ('STAGE 1: PROJECTED RATE - HZ', 'STAGE 2: LOCAL POOLED RATE - HZ',
                           'STAGE 3: NORMALIZED MT ACTIVITY - NOT HZ')
        for s, title in enumerate(titles):
            shift = s*305
            self.text(self.base, 20, 315+shift, title+' - CLOCKWISE FROM RIGHT')
            for d in range(8):
                self.text(self.base, 20+d*177, 345+shift, f'{d*45} DEG {view.NAMES[d]}', 1)
            maximum = activity_max if s == 2 else rate_max
            self.text(self.base, 20, 588+shift, 'FIXED COLOR SCALE: 0')
            bar = self.colors(np.linspace(0, maximum, 240), maximum)
            self.base[585+shift:605+shift, 290:530] = bar[None, :, :]
            self.text(self.base, 548, 588+shift, f'{maximum:g} '+('ACTIVITY' if s == 2 else 'HZ'))

    def text(self, canvas, x, y, message, scale=2, color=(230, 235, 245)):
        for char in message.upper():
            key = (char, scale)
            if key not in self.glyphs:
                glyph = view.FONT.get(char, '00000/'*6+'00000')
                mask = np.array([[bit == '1' for bit in row] for row in glyph.split('/')])
                self.glyphs[key] = np.repeat(np.repeat(mask, scale, axis=0), scale, axis=1)
            mask = self.glyphs[key]
            h, w = mask.shape
            if 0 <= x and x+w <= WIDTH and 0 <= y and y+h <= HEIGHT:
                canvas[y:y+h, x:x+w][mask] = color
            x += 6*scale

    def colors(self, values, maximum):
        v = np.clip(np.asarray(values, dtype=np.float64)/maximum, 0, 1)
        if self.palette == 'jet':
            rgb = np.clip(1.5-np.abs(4*v[..., None]-np.array([3, 2, 1])), 0, 1)
        else:
            rgb = np.clip(3*v[..., None]-np.array([0, 1, 2]), 0, 1)
        return np.rint(rgb*255).astype(np.uint8)

    def image(self, canvas, x, y, size, rgb):
        # Preserve rectangular images' aspect ratio and original x-right/y-down axes.
        h, w = rgb.shape[:2]
        scale = size/max(w, h)
        dw, dh = max(1, round(w*scale)), max(1, round(h*scale))
        resized = rgb[(np.arange(dh)*h//dh)[:, None], (np.arange(dw)*w//dw)[None, :]]
        left, top = x+(size-dw)//2, y+(size-dh)//2
        canvas[top:top+dh, left:left+dw] = resized

    def render(self, pixels, stages, index, total, dt, segment=None):
        canvas = self.base.copy()
        rgb = np.repeat(pixels[..., None], 3, axis=-1) if pixels.ndim == 2 else pixels
        self.image(canvas, 20, 56, 240, rgb)
        self.text(canvas, 295, 65, f'FRAME {index+1}/{total}  INPUT INDEX {index}', 3)
        self.text(canvas, 295, 103, f'INPUT TIME {index*dt:.2f}-{(index+1)*dt:.2f} S')
        expected = None
        if segment is None:
            label = 'INPUT: LABELS NOT PROVIDED'
            self.text(canvas, 295, 183, 'SPEED: NOT PROVIDED')
        else:
            moving = not segment['is_blank'] and segment['speed_px_per_frame'] > 0
            expected = segment['direction_slot_deg'] % 360 if moving else None
            label = ('INPUT: '+f'{expected:g} DEG' if moving else
                     'INPUT: BLANK' if segment['is_blank'] else 'INPUT: STATIC')
            self.text(canvas, 295, 183, f'SEGMENT {segment["segment_index"]}  SPEED {segment["speed_px_per_frame"]:g} PX/FRAME')
        self.text(canvas, 295, 141, label, 3, (255, 205, 70))
        for s, values in enumerate(stages):
            maximum = self.activity_max if s == 2 else self.rate_max
            shift = s*305
            means = values.mean(axis=(-2, -1), dtype=np.float64)
            winner = int(means.argmax()) if means.max() > 1e-8 else None
            rgb_maps = self.colors(values, maximum)
            for d, rgb_map in enumerate(rgb_maps):
                x = 20+d*177
                self.image(canvas, x, 368+shift, 160, rgb_map)
                self.text(canvas, x, 544+shift, f'MEAN {means[d]:.3f}')
                if expected is not None and math.isclose(expected, d*45, abs_tol=1e-6):
                    canvas[362+shift:366+shift, x:x+160] = (255, 205, 70)
                if winner == d:
                    canvas[530+shift:534+shift, x:x+160] = (60, 225, 245)
        return canvas


def render_recording(args, ffmpeg):
    data = load_recording(args.recording, args.stimulus, args.manifest, args.max_frames)
    output = Path(args.output)
    renderer = Renderer(data['info']['width'], data['info']['height'],
                        args.rate_max, args.activity_max, args.colormap)
    fps = args.playback_rate/data['dt']
    command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', f'{WIDTH}x{HEIGHT}', '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264',
               '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
               str(output/'mt_stages.mp4')]
    maps, stimulus = [], None
    process = None
    try:
        for binary, shape, _ in data['definitions']:
            maps.append(np.memmap(binary, mode='r', dtype='<f4', shape=tuple(shape)))
        shape = (data['info']['frames'], data['info']['height'], data['info']['width'])
        if data['info']['channels'] == 3:
            shape += (3,)
        stimulus = np.memmap(data['stimulus'], mode='r', dtype=np.uint8, offset=25, shape=shape)
        output.mkdir(parents=True, exist_ok=False)
        print(f'[start] {data["frames"]}/{data["info"]["frames"]} input frames; saved Python MT monitors; '
              f'{fps:g} video FPS; no model recomputation', flush=True)
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        started, seg_index = perf_counter(), 0
        fields = ['frame', 'input_index', 'start_ms', 'end_ms', 'segment', 'expected_deg', 'stage', 'units',
                  'winner_deg', 'spatial_min', 'spatial_max', 'fraction_at_or_above_color_max']
        fields += [f'mean_{d}_deg' for d in range(0, 360, 45)]
        with (output/'stage_statistics.csv').open('x', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for i in range(data['frames']):
                stages = [np.asarray(values[i])[data['order']] for values in maps]
                if any(not np.isfinite(v).all() or (v < 0).any() for v in stages):
                    raise ValueError(f'nonfinite/negative monitor values at input index {i}')
                segment = None
                if data['segments']:
                    while i*data['dt']*1000 >= data['segments'][seg_index]['end_ms']-1e-7:
                        seg_index += 1
                    segment = data['segments'][seg_index]
                canvas = renderer.render(stimulus[i], stages, i, data['frames'], data['dt'], segment)
                process.stdin.write(canvas.tobytes())
                if i in (0, data['frames']-1):
                    view.png(output/f'frame_{i+1:06d}.png', canvas.tobytes(), WIDTH, HEIGHT)
                expected = (segment['direction_slot_deg'] % 360 if segment and not segment['is_blank']
                            and segment['speed_px_per_frame'] > 0 else None)
                for s, values in enumerate(stages):
                    means = values.mean(axis=(-2, -1), dtype=np.float64)
                    maximum = args.activity_max if s == 2 else args.rate_max
                    row = dict(frame=i+1, input_index=i, start_ms=i*data['dt']*1000,
                               end_ms=(i+1)*data['dt']*1000,
                               segment=segment['segment_index'] if segment else None,
                               expected_deg=expected, stage=STAGES[s], units=UNITS[s],
                               winner_deg=int(means.argmax())*45 if means.max() > 1e-8 else None,
                               spatial_min=float(values.min()), spatial_max=float(values.max()),
                               fraction_at_or_above_color_max=float((values >= maximum).mean()))
                    row.update({f'mean_{d*45}_deg': float(means[d]) for d in range(8)})
                    writer.writerow(row)
                if args.progress_every and (i == 0 or (i+1) % args.progress_every == 0 or i+1 == data['frames']):
                    stream.flush()
                    elapsed = perf_counter()-started
                    print(f'[progress] {i+1}/{data["frames"]}, elapsed={elapsed:.1f}s, '
                          f'remaining~{elapsed/(i+1)*(data["frames"]-i-1):.1f}s', flush=True)
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError('ffmpeg failed; video is incomplete')
        config = dict(schema='python-mt-monitor-video-v1', complete=True, frames=data['frames'],
                      input_frames=data['info']['frames'], recorded_frames=data['report']['frames'],
                      complete_stimulus=data['frames'] == data['info']['frames'],
                      recording=str(data['root'].resolve()), stimulus=str(data['stimulus'].resolve()),
                      stimulus_sha256=data['stimulus_sha256'],
                      model_report_sha256=file_sha256(data['root']/'model_report.json'),
                      manifest_sha256=data['manifest_sha256'], segments_sha256=data['segments_sha256'],
                      monitor_metadata_sha256=[v[2] for v in data['definitions']],
                      stage_order=list(STAGES), output_direction_order_deg=list(range(0, 360, 45)),
                      frame_ms=data['dt']*1000, video_fps=fps, playback_rate=args.playback_rate,
                      rate_max=args.rate_max, activity_max=args.activity_max, colormap=args.colormap,
                      coordinates='x-right_y-down; clockwise from right',
                      pairing='input frame i and saved model response i; no time shift',
                      limitation='continuous model responses; not CARLsim MT spikes; startup history is retained')
        (output/'video_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
        print(f'[done] {data["frames"]} frames: {output}/mt_stages.mp4', flush=True)
        return config
    finally:
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait()
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
        for value in maps:
            value._mmap.close()
        if stimulus is not None:
            stimulus._mmap.close()
