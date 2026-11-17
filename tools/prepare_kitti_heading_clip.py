"""Prepare an indexed KITTI image clip with calibrated, timestamped references."""
import argparse
import csv
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from mstd.calibration import file_sha256, inspect_stimulus, write_stimulus
from mstd.egomotion_truth import camera_motion_reference
from mstd.v1_recording import iter_stimulus


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def append_registry(path, row):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        fields = csv.DictReader(stream).fieldnames
    with path.open('a', encoding='utf-8', newline='') as stream:
        csv.DictWriter(stream, fieldnames=fields).writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(r'E:\SLAM数据\KITTI'))
    parser.add_argument('--sequence', required=True, choices=[f'{i:02d}' for i in range(11)])
    parser.add_argument('--start', type=int, required=True)
    parser.add_argument('--frames', type=int, default=80)
    parser.add_argument('--frame-ms', type=int, default=104)
    args = parser.parse_args()
    if args.start < 0 or args.frames < 2 or args.frame_ms < 1:
        parser.error('positive cadence, at least two frames and nonnegative start required')
    encoder, probe = shutil.which('ffmpeg'), shutil.which('ffprobe')
    if not encoder or not probe:
        parser.error('ffmpeg and ffprobe are required for verified H.264 MP4 previews')

    sequence = args.source / 'data_odometry_gray/dataset/sequences' / args.sequence
    pose_path = args.source / 'KITTI_POSE/dataset/poses' / (args.sequence + '.txt')
    times_path, calib_path = sequence / 'times.txt', sequence / 'calib.txt'
    poses = np.loadtxt(pose_path).reshape(-1, 3, 4)
    times = np.loadtxt(times_path)
    end = args.start + args.frames - 1
    if len(poses) != len(times) or end >= len(times):
        parser.error('pose/time counts must agree and include the requested interval')
    long_reference = camera_motion_reference(poses[:, :, :3], poses[:, :, 3], times, 5)
    short_reference = camera_motion_reference(poses[:, :, :3], poses[:, :, 3], times, 1)
    p0_line = next(line.partition(':')[2] for line in calib_path.read_text().splitlines()
                   if line.startswith('P0:'))
    p0 = np.fromstring(p0_line, sep=' ').reshape(3, 4)

    target = times[args.start] + np.arange(args.frames) * args.frame_ms / 1000
    right = np.searchsorted(times, target).clip(1, len(times) - 1)
    indices = np.where(abs(times[right] - target) < abs(times[right - 1] - target), right, right - 1)
    if indices.tolist() != list(range(args.start, end + 1)):
        parser.error('this cadence would drop/duplicate source frames; choose a closer cadence')

    sid = (f'kitti{args.sequence}_f{args.start:06d}_f{end:06d}__160x48__'
           f'f{args.frames:06d}__dt{args.frame_ms}ms__gray_resize__v01')
    out = ROOT / 'input/stimuli/natural/kitti' / sid
    dat = out / (sid + '.dat')
    with (ROOT / 'input/registry.csv').open(encoding='utf-8-sig', newline='') as stream:
        known_inputs = list(csv.DictReader(stream))
    with (ROOT / 'experiments/registry.csv').open(encoding='utf-8-sig', newline='') as stream:
        known_artifacts = list(csv.DictReader(stream))
    if out.exists() or any(row['id'] == sid for row in known_artifacts):
        parser.error('stimulus already exists; preserve the existing version')
    if any(row['path'] == dat.relative_to(ROOT).as_posix() for row in known_inputs):
        parser.error('stimulus path already registered')

    frames, mapping, truth = [], [], []
    source_shape = None
    for frame, source_index in enumerate(indices):
        source_index = int(source_index)
        path = sequence / 'image_0' / f'{source_index:06d}.png'
        raw = path.read_bytes()
        with Image.open(io.BytesIO(raw)) as image:
            if source_shape is None:
                source_shape = image.size
            if image.size != source_shape:
                raise ValueError('source image geometry changes inside clip')
            frames.append(image.convert('L').resize((160, 48), Image.Resampling.LANCZOS).tobytes())
        mapping.append(dict(frame=frame, source_frame=source_index,
                            source_time_s=float(times[source_index]), target_time_s=float(target[frame]),
                            target_relative_time_s=frame * args.frame_ms / 1000,
                            time_offset_ms=float((times[source_index] - target[frame]) * 1000),
                            source_file=str(path), source_sha256=hashlib.sha256(raw).hexdigest()))
        position = poses[source_index, :, 3]
        next_index = min(source_index + 1, len(times) - 1)
        velocity = poses[source_index, :, :3].T @ (poses[next_index, :, 3] - position)
        if next_index == source_index:
            raise ValueError('a following GT pose is required for the forward-interval reference')
        velocity /= times[next_index] - times[source_index]
        truth.append(dict(
            frame=frame, input_time_s=frame * args.frame_ms / 1000,
            source_frame=source_index, source_time_s=float(times[source_index]),
            expected_heading_deg=float(long_reference['heading_x_deg'][source_index]),
            heading_short_window_deg=float(short_reference['heading_x_deg'][source_index]),
            heading_forward_interval_deg=float(np.degrees(np.arctan2(velocity[0], velocity[2]))),
            forward_interval_end_source_frame=next_index,
            vertical_bearing_deg=float(long_reference['bearing_y_deg'][source_index]),
            speed_mps=float(long_reference['speed_mps'][source_index]),
            velocity_camera_mps=long_reference['velocity_camera_mps'][source_index].tolist(),
            rotation_step_deg_s=float(long_reference['rotation_step_deg_s'][source_index]),
            rotation_window_deg_s=float(long_reference['rotation_window_deg_s'][source_index]),
            gt_reference_start_frame=int(long_reference['reference_start_index'][source_index]),
            gt_reference_end_frame=int(long_reference['reference_end_index'][source_index]),
            short_reference_start_frame=int(short_reference['reference_start_index'][source_index]),
            short_reference_end_frame=int(short_reference['reference_end_index'][source_index])))

    width, height = source_shape
    sx, sy = 160 / width, 48 / height
    intrinsics = dict(fx_px=float(p0[0, 0] * sx), fy_px=float(p0[1, 1] * sy),
                      cx_px=float((p0[0, 2] + .5) * sx - .5),
                      cy_px=float((p0[1, 2] + .5) * sy - .5))
    out.mkdir(parents=True, exist_ok=False)
    (out / 'preview').mkdir()
    write_stimulus(dat, 160, 48, frames)
    info = inspect_stimulus(dat)
    if info != dict(width=160, height=48, frames=args.frames, channels=1, type=4):
        raise AssertionError('native DAT header verification failed')
    for (frame, image), original in zip(iter_stimulus(dat), frames):
        if image.tobytes() != original:
            raise AssertionError(f'native DAT loader changed frame {frame}')
    with (out / 'frame_map.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(mapping[0]))
        writer.writeheader()
        writer.writerows(mapping)
    truth_name = sid + '.truth.jsonl'
    with (out / truth_name).open('x', encoding='utf-8') as stream:
        for row in truth:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    np.savetxt(out / 'camera0_to_world_poses.txt', poses[indices].reshape(-1, 12), fmt='%.12g')
    write_json(out / 'intrinsics.json', intrinsics)
    preview = out / 'preview/input.mp4'
    rate = Fraction(1000, args.frame_ms)
    subprocess.run([
        encoder, '-hide_banner', '-loglevel', 'error', '-n', '-f', 'rawvideo',
        '-pixel_format', 'gray', '-video_size', '160x48', '-framerate', str(rate), '-i', 'pipe:0',
        '-vf', 'scale=1280:384:flags=neighbor', '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-crf', '18', '-movflags', '+faststart', str(preview)], input=b''.join(frames), check=True)
    video = json.loads(subprocess.check_output([
        probe, '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=codec_name,nb_frames,avg_frame_rate,duration', '-of', 'json', str(preview)],
        text=True))['streams'][0]
    if (video['codec_name'] != 'h264' or int(video['nb_frames']) != args.frames
            or Fraction(video['avg_frame_rate']) != rate):
        raise AssertionError('MP4 verification failed')

    meta = dict(
        schema='natural-heading-stimulus-v1', stimulus_id=sid, stimulus=dat.name,
        stimulus_sha256=file_sha256(dat), **{k: info[k] for k in ('width', 'height', 'frames', 'channels')},
        frame_ms=args.frame_ms, intrinsics_file='intrinsics.json', intrinsics_px=intrinsics,
        truth_file=truth_name, frame_map='frame_map.csv', preview='preview/input.mp4',
        source_dataset=f'KITTI odometry {args.sequence}, rectified camera0 image_0',
        source_kind='external_read_only', source_directory=str(sequence), source_pose_file=str(pose_path),
        source_width=width, source_height=height, source_frame_start=args.start,
        source_frame_end_inclusive=end,
        resize=dict(method='Pillow LANCZOS', scale_x=sx, scale_y=sy,
                    pixel_center_map='u_new=(u_old+0.5)*scale-0.5', padding=False, cropping=False),
        timing=dict(method='nearest native exposure to fixed target grid; consecutive frames retained',
                    max_absolute_time_offset_ms=max(abs(row['time_offset_ms']) for row in mapping),
                    native_median_dt_ms=float(np.median(np.diff(times[indices])) * 1000),
                    source_first_to_last_s=float(times[end] - times[args.start]),
                    model_first_to_last_s=(args.frames - 1) * args.frame_ms / 1000,
                    model_presentation_duration_s=args.frames * args.frame_ms / 1000),
        truth_reference='camera translation heading atan2(vx,vz), x-right/y-down/z-forward; evaluation only',
        truth_windows=dict(expected_heading_deg='centered +/-5 native frames, legacy comparison',
                           heading_short_window_deg='centered +/-1 native frame',
                           heading_forward_interval_deg='current-to-next pose, current camera coordinates'),
        evaluation=dict(warmup_frames=20, constant_zero_baseline_required=True,
                        preserve_model_latency=True, reference_window_sensitivity_required=True),
        source_hashes={str(path): file_sha256(path) for path in (pose_path, times_path, calib_path)},
        generator_hashes={name: file_sha256(ROOT / name) for name in (
            'tools/prepare_kitti_heading_clip.py', 'src/mstd/egomotion_truth.py')},
        limitations=['Near-forward real-world clip; measured rotation and dynamic objects remain.',
                     'Selected using GT before model response. No accuracy acceptance has been established.',
                     'Centered GT references use future poses only during external evaluation.',
                     'Retain native exposure offsets and causal model latency when comparing curves.'])
    write_json(dat.with_suffix('.dat.json'), meta)
    artifact = dict(schema='mstd-artifact-v1', id=sid, kind='stimulus', status='completed',
                    stimulus_ids=[sid], input_paths=[str(sequence / 'image_0'), str(pose_path)],
                    input_sha256=meta['stimulus_sha256'], parent_run_ids=[],
                    notes='Continuous rectified KITTI images with resized camera0 calibration and external GT references.')
    write_json(out / 'artifact.json', artifact)
    write_json(out / 'verification.json', dict(native_info=info, payload_roundtrip=True, preview=video,
               files_sha256={path.relative_to(out).as_posix(): file_sha256(path)
                             for path in sorted(out.rglob('*')) if path.is_file()}))
    append_registry(ROOT / 'input/registry.csv', dict(path=dat.relative_to(ROOT).as_posix(),
                    width=160, height=48, frames=args.frames, channels=1,
                    sha256=meta['stimulus_sha256'], scope='reusable'))
    append_registry(ROOT / 'experiments/registry.csv', dict(id=sid, kind='stimulus',
                    path=out.relative_to(ROOT).as_posix(), status='completed', stimulus_ids=sid,
                    notes=artifact['notes']))
    print(json.dumps(dict(stimulus=dat.relative_to(ROOT).as_posix(),
                         max_timing_offset_ms=meta['timing']['max_absolute_time_offset_ms'],
                         preview=video), indent=2))


if __name__ == '__main__':
    main()
