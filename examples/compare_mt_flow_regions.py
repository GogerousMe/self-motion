"""Frozen Farneback reference vs saved MT directions; no model changes or GT fit."""
import argparse,csv,hashlib,json,math,subprocess,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
try:import cv2
except ImportError:
    sys.path.insert(0,str(Path(__file__).parent/'.deps'));import cv2

PARAMS=dict(pyr_scale=.5,levels=3,winsize=9,iterations=5,poly_n=5,poly_sigma=1.1,flags=0)
def flow(a,b,winsize):return cv2.calcOpticalFlowFarneback(a,b,None,**dict(PARAMS,winsize=winsize))
def reference(previous,current,winsize=9):
    h,w=current.shape;y,x=np.indices((h,w),dtype=np.float32)
    backward=flow(current,previous,winsize);forward=flow(previous,current,winsize)
    px=x+backward[:,:,0];py=y+backward[:,:,1]
    sampled=cv2.remap(forward,px,py,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    fb=np.linalg.norm(backward+sampled,axis=2)
    warped=cv2.remap(previous,px,py,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    photo=np.abs(current.astype(float)-warped.astype(float))
    texture=cv2.cornerMinEigenVal(current,blockSize=3,ksize=3)
    vector=-backward;mag=np.linalg.norm(vector,axis=2)
    keep=(px>=2)&(px<w-2)&(py>=2)&(py<h-2)&(x>=2)&(x<w-2)&(y>=2)&(y<h-2)
    keep&=(mag>=.25)&(mag<=10)&(fb<=.75)&(photo<=25)&(texture>=.0001)
    return vector,keep,fb

def sanity():
    rng=np.random.default_rng(20261110)
    a=cv2.GaussianBlur(rng.integers(0,256,(96,128),dtype=np.uint8),(3,3),.6)
    results=[]
    for dx,dy in [(2,0),(2,-2),(0,-2),(-2,-2),(-2,0),(-2,2),(0,2),(2,2)]:
        b=cv2.warpAffine(a,np.array([[1,0,dx],[0,1,dy]],np.float32),(128,96),borderMode=cv2.BORDER_REFLECT)
        v,mask,_=reference(a,b);mask[:12]=False;mask[-12:]=False;mask[:,:12]=False;mask[:,-12:]=False
        med=np.median(v[mask],axis=0);angle=float(np.degrees(np.arccos(np.clip(np.dot(med,[dx,dy])/np.linalg.norm(med)/math.hypot(dx,dy),-1,1))))
        assert mask.sum()>500 and angle<5 and np.linalg.norm(med-[dx,dy])<.2
        results.append(dict(dx=dx,dy=dy,median_vector=med.tolist(),angle_error_deg=angle,pixels=int(mask.sum())))
    v,mask,_=reference(a,a);assert not mask.any() and np.max(np.abs(v[12:-12,12:-12]))<.05
    return dict(eight_direction_translation_checks=results,identical_image_no_eligible_motion=True,
        identical_image_interior_max_component_px=float(np.max(np.abs(v[12:-12,12:-12]))),
        identical_image_full_max_component_px=float(np.max(np.abs(v))),seed=20261110)

def arrow(d,x,y,v,color,length=15):
    n=np.linalg.norm(v)
    if n<1e-12:return
    ux,uy=v/n;ex=x+length*ux;ey=y+length*uy
    d.line((x,y,ex,ey),fill=color,width=2)
    d.line((ex,ey,ex-5*ux+3*uy,ey-5*uy-3*ux),fill=color,width=2)
    d.line((ex,ey,ex-5*ux-3*uy,ey-5*uy+3*ux),fill=color,width=2)

def describe(values):
    if not len(values):return dict(n=0,mean_deg=None,median_deg=None,p90_deg=None,within_45_fraction=None)
    return dict(n=len(values),mean_deg=float(np.mean(values)),median_deg=float(np.median(values)),
        p90_deg=float(np.percentile(values,90)),within_45_fraction=float(np.mean(np.asarray(values)<=45)))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',required=True);p.add_argument('--recording',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.project_root).resolve();run=root/a.recording;out=Path(a.output).resolve()
    assert not out.exists();out.mkdir(parents=True)
    load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    report=load(run/'model_report.json');native=load(run/'artifact.json')
    sys.path.insert(0,str(root/'src'))
    from mstd.project_paths import resolve_stimulus
    from mstd.calibration import inspect_stimulus
    stim=resolve_stimulus(report['input']['input_path'],report['input']['stimulus_sha256'],root)
    info=inspect_stimulus(stim);n=report['frames'];w=info['width'];h=info['height'];dt=report['model']['frame_interval_s']
    assert report['complete'] and sha(stim)==report['input']['stimulus_sha256'] and (w,h,n)==(160,48,80)
    pixels=np.memmap(stim,mode='r',dtype='u1',offset=25,shape=(n,h,w))
    folder=run/'monitors/mt_readout_activity';meta=load(folder/'metadata.json')
    assert meta['complete'] and meta['shape']==[n,8,h,w] and meta['layout']=='TDYX' and meta['units']=='dimensionless'
    assert meta['coordinates']=='x-right_y-down' and meta['sample_indices']==list(range(n))
    assert np.allclose(meta['timestamps_s'],(np.arange(n)+1)*dt,atol=1e-8)
    assert (folder/'values.f32').stat().st_size==n*8*h*w*4
    activity=np.memmap(folder/'values.f32',mode='r',dtype='<f4',shape=(n,8,h,w))
    checks=sanity();(out/'reference_checks.json').write_text(json.dumps(checks,indent=2))
    angles=np.asarray(meta['directions_rad']);vectors=np.einsum('tdyx,dc->tyxc',activity,np.stack((np.cos(angles),np.sin(angles)),axis=1))
    mag=np.linalg.norm(vectors,axis=3);mt_keep=(mag>=.05)&(mag/np.maximum(activity.sum(axis=1),1e-12)>=.2)
    # Fixed image rectangles, not tracked or semantically segmented objects.
    regions=dict(left_background=(8,6,56,34),right_background=(104,6,152,34),road=(56,28,104,48))
    font=ImageFont.load_default(size=19);small=ImageFont.load_default(size=15)
    width,height=1536,1000;cellw=500;rows=[];aggregate={k:[] for k in regions};whole=[];coverage={k:[0,0,0,0] for k in regions}
    sensitivity={k:dict(reference_difference=[],stable_reference_mt_difference=[]) for k in regions}
    video=out/'mt_flow_regions.mp4'
    proc=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}',
        '-framerate',str(1/dt),'-i','pipe:0','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
    source_map=list(csv.DictReader((stim.parent/'frame_map.csv').open(encoding='utf-8-sig')))
    try:
        for i in range(n):
            image=np.asarray(pixels[i]);ref=np.zeros((h,w,2));ref_keep=np.zeros((h,w),bool)
            alt=ref.copy();alt_keep=ref_keep.copy()
            if i:
                ref,ref_keep,_=reference(np.asarray(pixels[i-1]),image)
                alt,alt_keep,_=reference(np.asarray(pixels[i-1]),image,winsize=15)
            ref_difference=np.degrees(np.arccos(np.clip((ref*alt).sum(axis=2)/np.maximum(np.linalg.norm(ref,axis=2)*np.linalg.norm(alt,axis=2),1e-12),-1,1)))
            dot=(vectors[i]*ref).sum(axis=2)/np.maximum(mag[i]*np.linalg.norm(ref,axis=2),1e-12)
            error=np.degrees(np.arccos(np.clip(dot,-1,1)));joint=mt_keep[i]&ref_keep
            canvas=Image.new('RGB',(width,height),(14,19,27));d=ImageDraw.Draw(canvas)
            d.text((18,8),f'KITTI04  frame {i}/79  t={i*dt:.3f}s  warmup={i<20}   Adjacent-frame reference, NOT ground truth',fill='white',font=font)
            d.text((18,35),'Cyan = MT; gold = Farneback. Same points in both arrow panels. Fixed arrow length = direction only.',fill='white',font=small)
            full=Image.fromarray(image).resize((960,288),Image.Resampling.NEAREST).convert('RGB');canvas.paste(full,(16,64));d=ImageDraw.Draw(canvas)
            for label,box in regions.items():
                x0,y0,x1,y1=box;color={'left_background':'#7bc8ff','right_background':'#ff9f80','road':'#b9f08a'}[label]
                d.rectangle((16+x0*6,64+y0*6,16+x1*6-1,64+y1*6-1),outline=color,width=2)
                d.text((16+x0*6,64+y0*6),label,fill=color,font=small)
            d.text((1000,85),'How to compare:',fill='white',font=font)
            for j,line in enumerate(['1. Follow each image rectangle.','2. Compare cyan/gold at same points.','3. Diverging arrows = disagreement.','4. No arrow = no eligible comparison.','Frame0 has no previous-frame flow.','No automatic time-lag fitting.','MT retains V1/neural temporal filtering.']):
                d.text((1000,120+27*j),line,fill='white',font=small)
            for ri,(label,(x0,y0,x1,y1)) in enumerate(regions.items()):
                left=16+ri*512;top=365;ww=x1-x0;hh=y1-y0
                rk=ref_keep[y0:y1,x0:x1];mk=mt_keep[i,y0:y1,x0:x1];jk=joint[y0:y1,x0:x1]
                values=error[y0:y1,x0:x1][jk];stats=describe(values)
                rows.append(dict(frame=i,region=label,warmup=i<20,area_pixels=ww*hh,reference_pixels=int(rk.sum()),mt_pixels=int(mk.sum()),joint_pixels=int(jk.sum()),**stats))
                if i>=20:
                    aggregate[label].extend(values.tolist());whole.extend(values.tolist())
                    coverage[label]=[v+q for v,q in zip(coverage[label],[ww*hh,int(rk.sum()),int(mk.sum()),int(jk.sum())])]
                    both=rk&alt_keep[y0:y1,x0:x1]
                    rd=ref_difference[y0:y1,x0:x1]
                    sensitivity[label]['reference_difference'].extend(rd[both].tolist())
                    stable_ref=both&(rd<=15)&mk
                    sensitivity[label]['stable_reference_mt_difference'].extend(error[y0:y1,x0:x1][stable_ref].tolist())
                fmt=lambda z:'--' if z is None else f'{z:.1f}'
                d.text((left,top),label,fill='white',font=font)
                d.text((left,top+26),f'paired pixels {int(jk.sum())}/{ww*hh}   median diff {fmt(stats["median_deg"])} deg',fill='white',font=small)
                crop=Image.fromarray(image[y0:y1,x0:x1]).convert('RGB');ch=round(hh*cellw/ww)
                # Original crop above two synchronized direction panels.
                raw=crop.resize((cellw,ch),Image.Resampling.NEAREST);canvas.paste(raw,(left,top+54))
                half=244;dh=round(hh*half/ww);cy=top+64+ch
                for ci,(field,color,title) in enumerate([(vectors[i],'#00ffe6','MT directions'),(ref,'#ffd200','Image-flow reference')]):
                    cx=left+ci*256;d=ImageDraw.Draw(canvas);d.text((cx,cy),title,fill=color,font=small)
                    pane=crop.resize((half,dh),Image.Resampling.NEAREST);pd=ImageDraw.Draw(pane)
                    for y in range(y0+2,y1,4):
                        for x in range(x0+2,x1,4):
                            if joint[y,x]:arrow(pd,(x-x0+.5)*half/ww,(y-y0+.5)*dh/hh,field[y,x],color)
                    canvas.paste(pane,(cx,cy+24));d=ImageDraw.Draw(canvas)
            d=ImageDraw.Draw(canvas)
            d.text((18,963),'ROIs are fixed image areas, NOT tree/road semantic masks. Reference filtering is heuristic, not a correctness guarantee.',fill='white',font=small)
            if i in (0,25,49,70):canvas.save(out/f'frame_{i:06d}.png')
            proc.stdin.write(canvas.tobytes())
    finally:proc.stdin.close();code=proc.wait()
    assert code==0
    summary={k:dict(**describe(v),eligible_reference_fraction=coverage[k][1]/coverage[k][0],mt_fraction=coverage[k][2]/coverage[k][0],joint_fraction=coverage[k][3]/coverage[k][0]) for k,v in aggregate.items()}
    summary['pooled_rois']=describe(whole)
    summary['sensitivity_winsize9_vs15']={k:{field:describe(v) for field,v in fields.items()} for k,fields in sensitivity.items()}
    settings=dict(schema='mt-flow-reference-v1',source_run=run.name,frames=n,frame_interval_s=dt,warmup_frames=20,
        opencv=cv2.__version__,numpy=np.__version__,farneback=PARAMS,regions_xyxy_exclusive=regions,
        reference='current->previous backward flow negated; forward-time displacement anchored at current pixel',
        reference_mask=dict(min_motion_px=.25,max_motion_px=10,fb_error_max_px=.75,photometric_error_max=25,min_corner_eigenvalue=.0001,border_px=2),
        mt_mask=dict(min_vector_strength=.05,min_resultant_ratio=.2),stats='pooled joint-eligible pixel-frame samples; no semantic masks',
        reference_sensitivity='independent winsize15 with same other settings/masks; unchanged primary9; conditional MT stats where both references eligible and differ<=15deg',
        timing='MT frame i vs image pair i-1,i; V1/neural temporal kernel retained; no lag optimization',
        display='fixed direction arrows at identical joint-eligible grid points, stride4; stats use all joint-eligible pixels',
        reference_is_ground_truth=False,source_frame_map='Original bundle/frame_map.csv; no source frames dropped')
    (out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'video_config.json').write_text(json.dumps(settings,indent=2))
    with (out/'direction_statistics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    artifact=dict(schema='mstd-artifact-v1',id=run.name+'/'+out.name,kind='visualization',status='completed',
        stimulus_ids=native['stimulus_ids'],input_paths=[stim.relative_to(root).as_posix()],input_sha256=sha(stim),parent_run_ids=[run.name],
        notes='MT vs independent image-flow reference; diagnostic, not ground truth or accuracy acceptance.',
        source_hashes={'examples/compare_mt_flow_regions.py':sha(Path(__file__)),'monitors/mt_readout_activity/values.f32':sha(folder/'values.f32')})
    (out/'artifact.json').write_text(json.dumps(artifact,indent=2))
    print(json.dumps(summary))

if __name__=='__main__':main()
