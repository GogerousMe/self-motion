"""Inspect saved V1 drives before MT, without changing or rerunning the model."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from mstd.calibration import file_sha256
from mstd.v1_recording import raw_info

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    run=Path(a.recording);manifest=Path(a.manifest);out=Path(a.output)
    config=json.loads((run/'model_config.json').read_text(encoding='utf-8'))
    report=json.loads((run/'model_report.json').read_text(encoding='utf-8'))
    info=raw_info(run/'monitors/v1');n,w,h=info['frames'],info['width'],info['height']
    meta=json.loads(manifest.read_text(encoding='utf-8'))
    assert info['provenance']['stimulus_sha256']==report['input']['stimulus_sha256']==meta['runs']['stimulus_sha256']
    assert file_sha256(manifest.parent/meta['runs']['stimulus'])==meta['runs']['stimulus_sha256']
    assert info['frame_interval_s']*1000==meta['frame_ms'] and n==meta['runs']['frames']
    projection_path=Path(config['projection_config'])
    projection=json.loads(projection_path.read_text(encoding='utf-8'))
    assert projection_path.name=='v1_projection.json'
    weights=np.array(projection['projections'][config['projection']]['weights'],np.float32)
    angles=np.asarray(projection['directions_rad']);scales=np.array(projection['scale_weights'],np.float32)
    v=np.memmap(run/'monitors/v1/rates.f32',dtype='<f4',mode='r',shape=(n,3,28,h,w))
    raw_mean=[];positive=[];negative=[];rect=[]
    for start in range(0,n,10):
        block=v[start:start+10];assert np.isfinite(block).all() and (block>=0).all()
        roi=block[:,:,:,4:-4,4:-4]
        raw_mean.extend(roi.mean(axis=(1,2,3,4),dtype=np.float64))
        scaled=np.einsum('tsfyx,s->tfyx',roi,scales,optimize=True)
        pos=np.einsum('tfyx,fd->tdyx',scaled,np.maximum(weights,0),optimize=True)
        neg=np.einsum('tfyx,fd->tdyx',scaled,np.maximum(-weights,0),optimize=True)
        positive.extend(pos.mean(axis=(2,3),dtype=np.float64))
        negative.extend(neg.mean(axis=(2,3),dtype=np.float64))
        rect.extend(np.maximum(pos-neg,0).mean(axis=(2,3),dtype=np.float64)*projection['gain'])
    raw_mean=np.asarray(raw_mean);positive=np.asarray(positive);negative=np.asarray(negative);rect=np.asarray(rect)
    summaries=[]
    for s in csv.DictReader((manifest.parent/meta['runs']['segments']).open(encoding='utf-8')):
        start,end=int(s['first_frame'])-1+20,int(s['last_frame']); ss=slice(start,end)
        avg=rect[ss].mean(axis=0);expected=float(s['expected_direction_deg']) if s['expected_direction_deg'] else None
        x=avg@np.cos(angles);y=avg@np.sin(angles)
        direction=float(np.degrees(np.arctan2(y,x))%360) if np.hypot(x,y)>1e-10 else None
        record=dict(segment=int(s['segment_index']),name=s['name'],expected=expected,
            v1_mean_hz=float(raw_mean[ss].mean()),positive_drive_mean=float(positive[ss].mean()),
            negative_drive_mean=float(negative[ss].mean()),rectified_projection_mean=float(avg.mean()),
            rectified_projection_peak=float(np.degrees(angles[avg.argmax()])%360) if avg.sum()>0 else None,
            projection_vector_deg=direction,projection_mean_by_channel=avg.tolist(),
            v1_first_last_stable_max_difference=float(np.max(np.abs(v[start]-v[end-1]))))
        if expected is not None:
            record['peak_error']=float(abs((record['rectified_projection_peak']-expected+180)%360-180))
        summaries.append(record)
    v._mmap.close()
    out.mkdir(parents=True,exist_ok=False)
    result=dict(input_sha256=meta['runs']['stimulus_sha256'],segments=summaries,
        caveat='Projection is a diagnostic linear readout; not equivalent to nonlinear delayed E/I circuit.')
    (out/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    artifact=dict(schema='mstd-artifact-v1',id=run.name+'/'+out.name,kind='visualization',status='completed',
        stimulus_ids=[manifest.parent.name+'__all_stimuli'],input_paths=[(manifest.parent/meta['runs']['stimulus']).as_posix()],
        input_sha256=result['input_sha256'],parent_run_ids=[run.name],notes=result['caveat'])
    (out/'artifact.json').write_text(json.dumps(artifact,indent=2),encoding='utf-8')
    root=Path(__file__).resolve().parents[1];registry=root/'experiments/registry.csv'
    with registry.open(encoding='utf-8-sig',newline='') as f:fields=next(csv.reader(f))
    with registry.open('a',encoding='utf-8',newline='') as f:
        csv.DictWriter(f,fieldnames=fields).writerow(dict(id=artifact['id'],kind='visualization',path=out.as_posix(),status='completed',parent_run_ids=run.name,stimulus_ids=artifact['stimulus_ids'][0],notes=artifact['notes']))
    for s in summaries:
        print(json.dumps({k:v for k,v in s.items() if k!='projection_mean_by_channel'}))

if __name__=='__main__':main()
