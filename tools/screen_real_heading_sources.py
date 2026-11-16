"""Read-only local KITTI/TUM inventory and GT-only low-rotation clip screening."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mstd.egomotion_truth import quaternion_xyzw_rotations,camera_motion_reference


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def candidates(reference,times,frames,min_speed):
    # Selection uses GT only, never model responses. Values are screening
    # limits, not acceptance thresholds for real-video accuracy.
    mask=((reference['speed_mps']>=min_speed)&(reference['velocity_camera_mps'][:,2]>0)
          &(abs(reference['heading_x_deg'])<=15)&(abs(reference['bearing_y_deg'])<=5)
          &(reference['rotation_window_deg_s']<=1)&(reference['rotation_step_deg_s']<=3))
    if len(mask)<frames:return []
    count=np.convolve(mask.astype(int),np.ones(frames,dtype=int),mode='valid')
    eligible=np.flatnonzero(count>=int(np.ceil(frames*.95)))
    choices=[]
    for start in eligible:
        s=slice(start,start+frames);rotation=reference['rotation_step_deg_s'][s]
        heading=reference['heading_x_deg'][s]
        choices.append(dict(start_frame=int(start),end_frame_inclusive=int(start+frames-1),
            start_time_s=float(times[start]),end_time_s=float(times[start+frames-1]),frames=frames,
            eligible_fraction=float(count[start]/frames),heading_mean_deg=float(heading.mean()),
            heading_min_deg=float(heading.min()),heading_max_deg=float(heading.max()),
            vertical_bearing_abs_max_deg=float(abs(reference['bearing_y_deg'][s]).max()),
            rotation_step_p95_deg_s=float(np.percentile(rotation,95)),rotation_step_max_deg_s=float(rotation.max()),
            rotation_window_mean_deg_s=float(reference['rotation_window_deg_s'][s].mean()),
            speed_mean_mps=float(reference['speed_mps'][s].mean()),
            dt_min_ms=float(np.diff(times[s]).min()*1000),dt_max_ms=float(np.diff(times[s]).max()*1000)))
    choices.sort(key=lambda r:(r['rotation_step_p95_deg_s'],abs(r['heading_mean_deg'])))
    selected=[]
    for row in choices:
        if all(row['end_frame_inclusive']<r['start_frame'] or row['start_frame']>r['end_frame_inclusive'] for r in selected):
            selected.append(row)
        if len(selected)==3:break
    return selected


def kitti_inventory(base):
    images=base/'data_odometry_gray/dataset/sequences';gt=base/'KITTI_POSE/dataset/poses'
    rows=[]
    for sequence in range(11):
        name=f'{sequence:02d}';directory=images/name;pose_path=gt/(name+'.txt')
        row=dict(dataset='kitti',sequence=name,images=str(directory/'image_0'),poses=str(pose_path))
        if not pose_path.is_file() or not (directory/'image_0').is_dir():
            rows.append({**row,'usable':False});continue
        pose=np.loadtxt(pose_path).reshape(-1,3,4)
        times_path=directory/'times.txt';times=np.loadtxt(times_path)
        count=sum(p.suffix.lower()=='.png' for p in (directory/'image_0').iterdir())
        calib=directory/'calib.txt'
        p0=next(line.partition(':')[2] for line in calib.read_text().splitlines() if line.startswith('P0:'))
        projection=np.fromstring(p0,sep=' ').reshape(3,4)
        row.update(usable=len(pose)==len(times)==count,image_count=count,pose_count=len(pose),
            duration_s=float(times[-1]-times[0]),nominal_interval_median_ms=float(np.median(np.diff(times))*1000),
            intrinsics_px=dict(fx_px=float(projection[0,0]),fy_px=float(projection[1,1]),cx_px=float(projection[0,2]),cy_px=float(projection[1,2])),
            source_hashes={str(p):sha(p) for p in (pose_path,times_path,calib)})
        if row['usable']:
            reference=camera_motion_reference(pose[:,:,:3],pose[:,:,3],times,5)
            row['candidates']=candidates(reference,times,80,1.)
            for clip in row['candidates']:
                clip['sequence']=name
                clip['first_image']=str(directory/'image_0'/f"{clip['start_frame']:06d}.png")
                clip['middle_image']=str(directory/'image_0'/f"{clip['start_frame']+40:06d}.png")
                clip['last_image']=str(directory/'image_0'/f"{clip['end_frame_inclusive']:06d}.png")
                assert all(Path(clip[k]).is_file() for k in ('first_image','middle_image','last_image'))
        rows.append(row)
    return rows


def tum_inventory(base):
    rows=[]
    for name in ('freiburg1_xyz','freiburg2_xyz'):
        path=base/'RGBD'/('rgbd_dataset_'+name+'.zip')
        if not path.is_file():continue
        with zipfile.ZipFile(path) as z:
            members=z.namelist()
            ground=next(n for n in members if n.endswith('/groundtruth.txt'))
            rgb=next(n for n in members if n.endswith('/rgb.txt'))
            ground_bytes=z.read(ground);rgb_bytes=z.read(rgb)
            poses=np.loadtxt(io.StringIO(ground_bytes.decode('utf-8-sig')))
            entries=[line.split() for line in rgb_bytes.decode('utf-8-sig').splitlines() if line and not line.startswith('#')]
            times=np.array([float(e[0]) for e in entries]);gt_times=poses[:,0]
            j=np.searchsorted(gt_times,times).clip(1,len(gt_times)-1)
            j=np.where(abs(gt_times[j]-times)<abs(gt_times[j-1]-times),j,j-1)
            difference=abs(gt_times[j]-times)
            r=quaternion_xyzw_rotations(poses[j,4:8]);position=poses[j,1:4]
            half=max(1,int(round(.5/np.median(np.diff(times)))))
            ref=camera_motion_reference(r,position,times,half)
            clips=candidates(ref,times,80,.02)
            for clip in clips:
                s=clip['start_frame'];e=clip['end_frame_inclusive']
                clip.update(first_rgb_member=rgb.rpartition('/')[0]+'/'+entries[s][1],
                    middle_rgb_member=rgb.rpartition('/')[0]+'/'+entries[s+40][1],last_rgb_member=rgb.rpartition('/')[0]+'/'+entries[e][1],
                    gt_nearest_match_max_offset_ms=float(difference[s:e+1].max()*1000))
                assert clip['first_rgb_member'] in members
            rows.append(dict(dataset='tum',sequence=name,archive=str(path),image_count=len(times),pose_count=len(poses),
                nominal_interval_median_ms=float(np.median(np.diff(times))*1000),duration_s=float(times[-1]-times[0]),
                poses_member=ground,rgb_list_member=rgb,candidates=clips,
                source_metadata_sha256=dict(groundtruth=hashlib.sha256(ground_bytes).hexdigest(),rgb_list=hashlib.sha256(rgb_bytes).hexdigest()),
                note='Nearest pose associations for screening only; timestamps/undistortion need explicit handling before model evaluation.'))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kitti',default=r'E:\SLAM数据\KITTI');p.add_argument('--tum',default=r'E:\SLAM数据\TUM')
    p.add_argument('--output',required=True);a=p.parse_args()
    out=(ROOT/a.output).resolve()
    if not out.is_relative_to(ROOT/'experiments/validation'):p.error('use experiments/validation')
    out.mkdir(parents=True,exist_ok=False)
    kitti=kitti_inventory(Path(a.kitti));tum=tum_inventory(Path(a.tum))
    ranked=[{**clip,'dataset':'kitti'} for r in kitti for clip in r.get('candidates',[])]
    ranked.sort(key=lambda r:r['rotation_step_p95_deg_s'])
    result=dict(kitti=kitti,tum=tum,kitti_top_candidates=ranked[:6],
        criteria=dict(frames=80,min_forward_speed_kitti_mps=1.,min_forward_speed_tum_mps=.02,
            max_horizontal_bearing_deg=15,max_vertical_bearing_deg=5,max_rotation_window_deg_s=1,
            max_rotation_step_deg_s=3,min_eligible_fraction=.95),
        reference='camera velocity = R_camera_to_world.T * centered world displacement / dt; heading = atan2(vx,vz)',
        source_hashes={p:sha(ROOT/p) for p in ('tools/screen_real_heading_sources.py','src/mstd/egomotion_truth.py')},
        limitations=['GT-only candidate screening, no model accuracy result.',
            'KITTI uses roughly1s centered displacement direction references; TUM roughly1s, nearest pose match for screening.',
            'At least95% of frames meet screening limits; max/percentile values retained, not guaranteed zero rotation.',
            'Native timestamps are not assumed uniformly50ms. No images or source archives were modified.'])
    (out/'inventory.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    artifact=dict(schema='mstd-artifact-v1',id=out.name,kind='validation',status='completed',stimulus_ids=[],
        input_paths=[str(Path(a.kitti)),str(Path(a.tum))],parent_run_ids=[],
        notes='Read-only external real-source inventory and GT-only selection; no model run or real-video acceptance.')
    (out/'artifact.json').write_text(json.dumps(artifact,indent=2),encoding='utf-8')
    registry=ROOT/'experiments/registry.csv'
    with registry.open(encoding='utf-8-sig',newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;assert not any(r['id']==out.name for r in reader)
    with registry.open('a',encoding='utf-8',newline='') as f:csv.DictWriter(f,fieldnames=fields).writerow(dict(id=out.name,kind='validation',path=out.relative_to(ROOT).as_posix(),status='completed',notes=artifact['notes']))
    print(json.dumps(dict(kitti_sequences=len(kitti),usable_kitti=sum(r['usable'] for r in kitti),kitti_candidates=len(ranked),tum_candidates={r['sequence']:len(r['candidates']) for r in tum},best_kitti=ranked[:3]),ensure_ascii=False))


if __name__=='__main__':main()
