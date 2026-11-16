"""Prepare one inspected KITTI04 pilot, with explicit cadence and calibration."""
import argparse,csv,hashlib,io,json
from pathlib import Path
import shutil,subprocess,sys
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mstd.calibration import write_stimulus,inspect_stimulus
from mstd.egomotion_truth import camera_motion_reference

SID='kitti04_f000153_f000232__160x48__f000080__dt104ms__gray_resize__v01'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default=r'E:\SLAM数据\KITTI')
    p.add_argument('--output',default='input/stimuli/natural/kitti/'+SID)
    a=p.parse_args();base=Path(a.source);out=(ROOT/a.output).resolve()
    if not out.is_relative_to(ROOT/'input/stimuli/natural/kitti'):p.error('use input/stimuli/natural/kitti')
    encoder=shutil.which('ffmpeg');probe=shutil.which('ffprobe')
    if not encoder or not probe:p.error('ffmpeg/ffprobe required for verified MP4 preview')
    sequence=base/'data_odometry_gray/dataset/sequences/04'
    pose_path=base/'KITTI_POSE/dataset/poses/04.txt';times_path=sequence/'times.txt';calib_path=sequence/'calib.txt'
    pose=np.loadtxt(pose_path).reshape(-1,3,4);times=np.loadtxt(times_path)
    assert len(pose)==len(times)
    reference=camera_motion_reference(pose[:,:,:3],pose[:,:,3],times,5)
    p0=np.fromstring(next(s.partition(':')[2] for s in calib_path.read_text().splitlines() if s.startswith('P0:')),sep=' ').reshape(3,4)
    target=times[153]+np.arange(80)*.104
    right=np.searchsorted(times,target).clip(1,len(times)-1)
    indices=np.where(abs(times[right]-target)<abs(times[right-1]-target),right,right-1)
    assert indices.tolist()==list(range(153,233))  # No hidden drops or duplicates for this inspected pilot.
    with (ROOT/'input/registry.csv').open(encoding='utf-8-sig',newline='') as f:reader=csv.DictReader(f);input_fields=reader.fieldnames;index=list(reader)
    with (ROOT/'experiments/registry.csv').open(encoding='utf-8-sig',newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;registered=list(reader)
    if any(r['id']==out.name for r in registered):p.error('stimulus ID already registered')
    out.mkdir(parents=True,exist_ok=False);(out/'preview').mkdir()
    frames=[];mapping=[];truth=[];source_shape=None
    for i,k in enumerate(indices):
        path=sequence/'image_0'/f'{k:06d}.png';raw=path.read_bytes()
        image=Image.open(io.BytesIO(raw)).convert('L')
        if source_shape is None:source_shape=image.size
        assert source_shape==image.size
        frames.append(image.resize((160,48),Image.Resampling.LANCZOS).tobytes())
        mapping.append(dict(frame=i,source_frame=int(k),source_time_s=float(times[k]),target_time_s=float(target[i]),
            target_relative_time_s=float(i*.104),time_offset_ms=float((times[k]-target[i])*1000),
            source_file=str(path),source_sha256=hashlib.sha256(raw).hexdigest()))
        truth.append(dict(frame=i,input_time_s=float(i*.104),source_frame=int(k),source_time_s=float(times[k]),
            expected_heading_deg=float(reference['heading_x_deg'][k]),vertical_bearing_deg=float(reference['bearing_y_deg'][k]),
            speed_mps=float(reference['speed_mps'][k]),rotation_step_deg_s=float(reference['rotation_step_deg_s'][k]),
            rotation_window_deg_s=float(reference['rotation_window_deg_s'][k]),
            gt_reference_start_frame=int(reference['reference_start_index'][k]),gt_reference_end_frame=int(reference['reference_end_index'][k])))
    width,height=source_shape;sx,sy=160/width,48/height
    intrinsics=dict(fx_px=float(p0[0,0]*sx),fy_px=float(p0[1,1]*sy),
                    cx_px=float((p0[0,2]+.5)*sx-.5),cy_px=float((p0[1,2]+.5)*sy-.5))
    path=out/(out.name+'.dat');write_stimulus(path,160,48,frames)
    assert inspect_stimulus(path)==dict(width=160,height=48,frames=80,channels=1,type=4)
    with (out/'frame_map.csv').open('x',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(mapping[0]));w.writeheader();w.writerows(mapping)
    truth_name=out.name+'.truth.jsonl'
    with (out/truth_name).open('x',encoding='utf-8') as f:
        for t in truth:f.write(json.dumps(t,allow_nan=False)+'\n')
    (out/'intrinsics.json').write_text(json.dumps(intrinsics,indent=2),encoding='utf-8')
    preview=out/'preview/input.mp4'
    subprocess.run([encoder,'-hide_banner','-loglevel','error','-n','-f','rawvideo','-pixel_format','gray',
        '-video_size','160x48','-framerate','125/13','-i','pipe:0','-vf','scale=1280:384:flags=neighbor',
        '-an','-c:v','libx264','-pix_fmt','yuv420p','-crf','18','-movflags','+faststart',str(preview)],input=b''.join(frames),check=True)
    video=json.loads(subprocess.check_output([probe,'-v','error','-select_streams','v:0','-show_entries','stream=codec_name,nb_frames,avg_frame_rate,duration','-of','json',str(preview)],text=True))['streams'][0]
    assert video['codec_name']=='h264' and video['nb_frames']=='80' and video['avg_frame_rate']=='125/13'
    meta=dict(schema='natural-heading-stimulus-v1',stimulus_id=out.name,stimulus=path.name,stimulus_sha256=sha(path),
        width=160,height=48,frames=80,channels=1,frame_ms=104,intrinsics_file='intrinsics.json',intrinsics_px=intrinsics,
        truth_file=truth_name,frame_map='frame_map.csv',preview='preview/input.mp4',
        source_dataset='KITTI odometry 04, rectified left camera image_0',source_kind='external_read_only',
        source_directory=str(sequence),source_pose_file=str(pose_path),source_width=width,source_height=height,
        source_frame_start=153,source_frame_end_inclusive=232,
        resize=dict(method='Pillow LANCZOS',scale_x=sx,scale_y=sy,pixel_center_map='u_new=(u_old+0.5)*scale-0.5',padding=False,cropping=False),
        timing=dict(method='nearest native exposure to fixed104ms target grid; all80 source frames retained',
            max_absolute_time_offset_ms=max(abs(r['time_offset_ms']) for r in mapping),native_median_dt_ms=float(np.median(np.diff(times[153:233]))*1000)),
        truth_reference='R_i.T*(p_i+5-p_i-5)/dt: roughly1s centered translation reference, external evaluation only',
        source_hashes={str(p):sha(p) for p in (pose_path,times_path,calib_path)},
        generator_hashes={s:sha(ROOT/s) for s in ('tools/prepare_kitti04_heading_pilot.py','src/mstd/egomotion_truth.py')},
        limitations=['Exploratory real-image pilot; not exact pure translation, not official KITTI SLAM evaluation.',
            'Dynamic scene content and temporal filtering remain; no real-video accuracy claim before model run.',
            '104ms and transformed calibration are required; do not run as50ms/80x80.'])
    path.with_suffix('.dat.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    artifact=dict(schema='mstd-artifact-v1',id=out.name,kind='stimulus',status='completed',stimulus_ids=[out.name],
        input_paths=[str(sequence/'image_0'),str(pose_path)],input_sha256=meta['stimulus_sha256'],parent_run_ids=[],
        notes='External KITTI04 pilot153..232,160x48,104ms; image input only, camera poses used only as evaluation reference.')
    (out/'artifact.json').write_text(json.dumps(artifact,indent=2),encoding='utf-8')
    with (ROOT/'input/registry.csv').open('a',encoding='utf-8',newline='') as f:csv.DictWriter(f,fieldnames=input_fields).writerow(dict(path=path.relative_to(ROOT).as_posix(),width=160,height=48,frames=80,channels=1,sha256=meta['stimulus_sha256'],scope='reusable'))
    with (ROOT/'experiments/registry.csv').open('a',encoding='utf-8',newline='') as f:csv.DictWriter(f,fieldnames=fields).writerow(dict(id=out.name,kind='stimulus',path=out.relative_to(ROOT).as_posix(),status='completed',stimulus_ids=out.name,notes=artifact['notes']))
    print(json.dumps(dict(output=str(out),source_shape=source_shape,intrinsics=intrinsics,max_timing_offset_ms=meta['timing']['max_absolute_time_offset_ms'],preview=video)))


if __name__=='__main__':main()
