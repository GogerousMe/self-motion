"""Render paired observed MT spike rates with one shared Hz color scale."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'examples'))

from mstd.calibration import file_sha256
from mstd.v1_recording import dump_json, iter_stimulus
import visualize_mt_gratings as drawing
from visualize_python_spiking import close_recording, load_recording, paired_stimulus
from run_mt_spatial_controls import register_run, VARIANT_SWITCHES


def render(run, case_name, maximum, variants=('baseline', 'spatial_only')):
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError('positive finite rate maximum required')
    folder = run / case_name
    case = json.loads((folder / 'case.json').read_text())
    stimulus = folder / 'stimulus.dat'
    if not stimulus.is_file():
        raise ValueError('this renderer requires an image fixture from the static/texture cases')
    encoder = shutil.which('ffmpeg')
    if encoder is None:
        raise RuntimeError('install ffmpeg to write MP4')
    recordings = []
    try:
        for variant in variants:
            data = load_recording(folder / variant)
            recordings.append(data)
            paired_stimulus(data, stimulus)
            if data['indices'] != list(range(data['report']['frames'])):
                raise ValueError('every-frame recording required')
        if any(data['indices'] != recordings[0]['indices'] for data in recordings):
            raise ValueError('paired frame indices disagree')
        dt = recordings[0]['report']['model']['frame_interval_s']
        if any(data['report']['model']['frame_interval_s'] != dt for data in recordings):
            raise ValueError('paired frame intervals disagree')
        identifier = run.name + '__' + case_name + '__rates__v01'
        category = 'mt_competition' if len(variants) == 4 else 'mt_spatial'
        output = ROOT / 'outputs/comparisons' / category / identifier
        output.mkdir(parents=True, exist_ok=False)
        artifact = dict(schema='mstd-artifact-v1', id=identifier, kind='comparison', status='running',
            parent_run_ids=[json.loads((folder / variant / 'artifact.json').read_text())['id']
                            for variant in variants], stimulus_ids=[],
            input_paths=[stimulus.relative_to(ROOT).as_posix()], input_sha256=file_sha256(stimulus),
            source_hashes={'tools/render_mt_spatial_controls.py': file_sha256(Path(__file__)),
                           'examples/visualize_mt_gratings.py': file_sha256(ROOT / 'examples/visualize_mt_gratings.py')},
            notes='Observed excitatory spikes/frame duration, Hz. Shared color scale; nearest-neighbor display.')
        frames = len(recordings[0]['indices'])
        drawing.H = 240 + len(variants) * 200
        command = [encoder, '-v', 'error', '-n', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                   '-s', f'1440x{drawing.H}', '-r', str(1 / dt), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
                   '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output / 'mt_spatial_comparison.mp4')]
        process = None
        try:
            with (output / 'ffmpeg.log').open('x') as log:
                process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
                for index, pixels in iter_stimulus(stimulus, frames):
                    canvas = bytearray(bytes((15, 19, 27)) * drawing.W * drawing.H)
                    drawing.text(canvas, 24, 10, f'FRAME {index} / {frames - 1}   INPUT {index * dt:.3f} S', 2)
                    gray = pixels if pixels.ndim == 2 else pixels[:, :, 0]
                    drawing.image(canvas, 24, 38, 192, gray.reshape(-1), gray.shape[1], gray.shape[0])
                    title = ('ROWS BASELINE / SPATIAL / COMPETITION / BOTH' if len(variants) == 4
                             else 'BASELINE TOP - SPATIAL ONLY BOTTOM')
                    drawing.text(canvas, 260, 45, title, 2)
                    drawing.text(canvas, 260, 85, 'OBSERVED MT EXCITATORY SPIKE RATE - HZ', 2)
                    if case.get('direction_deg') is not None:
                        angle = case['direction_deg']
                        drawing.arrow(canvas, 280, 140, angle, 40)
                        drawing.text(canvas, 350, 135, f'INPUT {angle} DEG   SPEED {case["speed"]:g} PX/FRAME', 2)
                    else:
                        drawing.text(canvas, 260, 135, 'STATIC TEXTURE - SAME IMAGE REPEATED', 2)
                    drawing.text(canvas, 260, 175, f'SHARED COLOR SCALE 0 TO {maximum:g} HZ', 2)
                    for x in range(400):
                        drawing.rect(canvas, 260 + x, 200, 1, 12, drawing.color(x / 399, 1, 'jet'))
                    for row, data in enumerate(recordings):
                        values = data['saved']['mt_exc_rates'][1][index][data['order']]
                        for direction, rates in enumerate(values):
                            x = 24 + direction * 174
                            y = 260 + row * 200
                            drawing.text(canvas, x, y - 18, f'{direction * 45:03d} {drawing.NAMES[direction]}',
                                         2 if len(variants) == 4 else 1)
                            drawing.image(canvas, x, y, 160, rates.reshape(-1), rates.shape[1], rates.shape[0], maximum, 'jet')
                    if index in (0, frames - 1):
                        drawing.png(output / f'frame_{index:04d}.png', canvas, width=drawing.W, height=drawing.H)
                    process.stdin.write(canvas)
                process.stdin.close()
                if process.wait() != 0:
                    raise RuntimeError('MP4 encoding failed; see ffmpeg.log')
            dump_json(output / 'visualization.json', dict(complete=True, frames=frames, frame_interval_s=dt,
                unit='Hz', rate_max=maximum, direction_order_deg=list(range(0, 360, 45)),
                coordinates='x-right_y-down; clockwise from right', response_timestamp='frame_end',
                input_timestamp='frame_start', row_order=list(variants),
                monitor='mt_exc_rates', stimulus_sha256=artifact['input_sha256']))
            artifact['status'] = 'completed'
        except BaseException:
            artifact['status'] = 'failed'
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
            raise
        finally:
            register_run(output, artifact)
        print(output)
    finally:
        for data in recordings:
            close_recording(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--case', default='texture_d000_s1')
    parser.add_argument('--rate-max', type=float, default=300)
    parser.add_argument('--four-way', action='store_true')
    args = parser.parse_args()
    run = args.run.resolve()
    if not run.is_relative_to(ROOT.resolve()):
        raise ValueError('run must be inside the project')
    if args.case not in {path.name for path in run.iterdir() if path.is_dir()}:
        raise ValueError('case must name one case folder inside the run')
    variants = tuple(VARIANT_SWITCHES) if args.four_way else ('baseline', 'spatial_only')
    render(run, args.case, args.rate_max, variants)


if __name__ == '__main__':
    main()
