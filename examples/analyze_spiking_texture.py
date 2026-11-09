"""Quantitative inspection of recorded Python MT spikes, not a re-simulation.

Writes only to a new view folder. Model thresholds are diagnostic, not fitted.
"""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--output',required=True);p.add_argument('--border',type=int,default=4)
    a=p.parse_args(); root=Path(__file__).resolve().parents[1]
    run=Path(a.recording);manifest=Path(a.manifest);out=Path(a.output)
    r=json.loads((run/'model_report.json').read_text(encoding='utf-8'))
    m=json.loads(manifest.read_text(encoding='utf-8'));w=r['model']['width'];h=r['model']['height'];n=r['frames'];dt=r['model']['frame_interval_s']
    stimulus=manifest.parent/m['runs']['stimulus'];digest=hashlib.sha256(stimulus.read_bytes()).hexdigest()
    assert r['complete'] and r['model']['mt_backend']=='python_spiking'
    assert digest==m['runs']['stimulus_sha256']==r['input']['stimulus_sha256']
    assert n==m['runs']['frames'] and dt*1000==m['frame_ms']
    assert 0<=a.border<min(w,h)/2
    segments=list(csv.DictReader((manifest.parent/m['runs']['segments']).open(encoding='utf-8')))
    layers={};checks={};angles=None
    for name in ('mt_exc_rates','mt_inh_rates','mt_filtered_rates','mt_readout_activity'):
        directory=run/'monitors'/name;meta=json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
        assert meta['complete'] and meta['shape']==[n,8,h,w] and meta['sample_indices']==list(range(n))
        assert np.allclose(meta['timestamps_s'],(np.arange(n)+1)*dt)
        assert (directory/'values.f32').stat().st_size==n*8*h*w*4
        angles=np.asarray(meta.get('directions_rad',angles))
        values=np.memmap(directory/'values.f32',mode='r',dtype='<f4',shape=(n,8,h,w))
        means=[];local=[];finite=True;minimum=float('inf');maximum=0.
        b=a.border;roi=(slice(b,h-b),slice(b,w-b))
        for start in range(0,n,20):
            block=values[start:start+20]
            finite=finite and bool(np.isfinite(block).all());minimum=min(minimum,float(block.min()));maximum=max(maximum,float(block.max()))
            v=block[:,:,roi[0],roi[1]].astype(np.float64)
            means.extend(v.mean(axis=(2,3)))
            if name=='mt_readout_activity':
                x=np.einsum('tdyx,d->tyx',v,np.cos(angles));y=np.einsum('tdyx,d->tyx',v,np.sin(angles))
                local.extend(np.hypot(x,y).mean(axis=(1,2)))
        values._mmap.close();assert finite and minimum>=0
        checks[name]=dict(finite=True,min=minimum,max=maximum)
        layers[name]=np.asarray(means)
        if local:local=np.asarray(local);layers['local_evidence']=local
    means=layers['mt_exc_rates'];total=means.sum(axis=1)
    vx=means@np.cos(angles);vy=means@np.sin(angles)
    direction=np.degrees(np.arctan2(vy,vx))%360
    resultant=np.divide(np.hypot(vx,vy),total,out=np.zeros(n),where=total>0)
    activity=layers['mt_readout_activity'];avx=activity@np.cos(angles);avy=activity@np.sin(angles)
    adir=np.degrees(np.arctan2(avy,avx))%360
    summaries=[];frame_rows=[]
    for s in segments:
        first,last=int(s['first_frame'])-1,int(s['last_frame']);ix=np.arange(first,last);stable=ix[ix>=first+20]
        expected=float(s['expected_direction_deg']) if s['expected_direction_deg'] else None
        ss=stable;avg=means[ss].mean(axis=0);x=avg@np.cos(angles);y=avg@np.sin(angles)
        summary=dict(segment=int(s['segment_index']),name=s['name'],phase=s['phase'],expected=expected,
            stable_bins=len(ss),e_mean_hz=float(avg.mean()),e_total_hz=float(avg.sum()),
            i_mean_hz=float(layers['mt_inh_rates'][ss].mean()),
            peak_deg=float(np.degrees(angles[avg.argmax()])%360) if avg.sum()>0 else None,
            vector_deg=float(np.degrees(np.arctan2(y,x))%360) if np.hypot(x,y)>1e-12 else None,
            resultant=float(np.hypot(x,y)/avg.sum()) if avg.sum()>0 else 0.,
            local_evidence_mean=float(layers['local_evidence'][ss].mean()),
            activity_global_vector_mean=float(np.hypot(avx[ss],avy[ss]).mean()))
        if expected is not None:
            err=np.abs((direction[ix]-expected+180)%360-180);ae=np.abs((adir[ix]-expected+180)%360-180)
            summary.update(frame_mae=float(err[20:].mean()),frame_p95=float(np.percentile(err[20:],95)),
                activity_mae=float(ae[20:].mean()),peak_error=float(abs((summary['peak_deg']-expected+180)%360-180)),
                fraction_direction_error_over_22p5=float((err[20:]>22.5).mean()))
            good=(err<=22.5)&(resultant[ix]>=.1)&(total[ix]>0)
            start=next((i for i in range(len(ix)-4) if good[i:i+5].all()),None)
            summary['first_good_bin_end_ms']=float((start+1)*dt*1000) if start is not None else None
        summaries.append(summary)
        for i in ix:
            row=dict(frame=int(i),end_s=(i+1)*dt,segment=summary['segment'],phase=s['phase'],expected=expected,
                e_mean_hz=float(means[i].mean()),i_mean_hz=float(layers['mt_inh_rates'][i].mean()),
                direction_deg=float(direction[i]) if total[i]>0 else None,resultant=float(resultant[i]),
                local_evidence=float(layers['local_evidence'][i]),activity_direction_deg=float(adir[i]),
                activity_global_magnitude=float(np.hypot(avx[i],avy[i])))
            frame_rows.append(row)
    out.mkdir(parents=True,exist_ok=False)
    with (out/'frame_metrics.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(frame_rows[0]));writer.writeheader();writer.writerows(frame_rows)
    result=dict(input_sha256=digest,roi_border=a.border,exclusion_ms=1000,monitors=checks,segments=summaries,
        caveats=['One seed; static angles are undefined, displayed numerical vectors are not motion truth.',
                 '22.5deg and resultant 0.1 are diagnostic screens, not a preregistered new-model acceptance threshold.'])
    (out/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    artifact=dict(schema='mstd-artifact-v1',id=run.name+'/texture_metrics__v01',kind='visualization',status='completed',
        stimulus_ids=[manifest.parent.name+'__all_stimuli'],input_paths=[str(stimulus)],input_sha256=digest,
        parent_run_ids=[run.name],notes='Post-run quantitative inspection; no model changes.')
    (out/'artifact.json').write_text(json.dumps(artifact,indent=2),encoding='utf-8')
    registry=root/'experiments/registry.csv'
    with registry.open(encoding='utf-8-sig',newline='') as f:fields=next(csv.reader(f))
    with registry.open('a',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writerow(dict(id=artifact['id'],kind='visualization',path=out.as_posix(),status='completed',parent_run_ids=run.name,stimulus_ids=artifact['stimulus_ids'][0],notes=artifact['notes']))
    for s in summaries: print(json.dumps(s))

if __name__=='__main__':main()
