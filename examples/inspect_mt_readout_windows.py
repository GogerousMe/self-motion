"""Dense pixel directions and common-mask readout comparison, no neural rerun."""
import argparse,csv,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
try:from compare_mt_flow_regions import reference,describe
except ImportError:
    sys.path.insert(0,str(Path(__file__).parents[1]/'examples'))
    from compare_mt_flow_regions import reference,describe

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',required=True);p.add_argument('--recording',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.project_root);run=root/a.recording;out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    load=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
    r=load(run/'model_report.json');art=load(run/'artifact.json');w=r['model']['width'];h=r['model']['height'];n=r['frames'];dt=r['model']['frame_interval_s']
    sys.path.insert(0,str(root/'src'));from mstd.project_paths import resolve_stimulus
    stim=resolve_stimulus(r['input']['input_path'],r['input']['stimulus_sha256'],root)
    assert hashlib.sha256(stim.read_bytes()).hexdigest()==r['input']['stimulus_sha256']
    images=np.memmap(stim,mode='r',dtype='u1',offset=25,shape=(n,h,w))
    arrays=[]
    for name,unit in [('mt_readout_activity','dimensionless'),('mt_exc_rates','Hz')]:
        folder=run/'monitors'/name;m=load(folder/'metadata.json')
        assert m['complete'] and m['shape']==[n,8,h,w] and m['units']==unit and m['layout']=='TDYX'
        assert m['sample_indices']==list(range(n)) and np.allclose(m['timestamps_s'],(np.arange(n)+1)*dt)
        assert (folder/'values.f32').stat().st_size==n*8*h*w*4
        arrays.append(np.memmap(folder/'values.f32',mode='r',dtype='<f4',shape=(n,8,h,w)))
    assert (w,h,n)==(160,48,80)
    angle=np.array(m['directions_rad']);units=np.stack((np.cos(angle),np.sin(angle)),axis=1)
    regions={'left_background':(8,6,56,34),'right_background':(104,6,152,34),'road':(56,28,104,48)}
    allerr={k:[[],[]] for k in regions};rows=[]
    def field(x):
        v=np.einsum('dyx,dc->yxc',x,units);mag=np.linalg.norm(v,axis=2)
        return v,(mag>=.05)&(mag/np.maximum(x.sum(axis=0),1e-12)>=.2)
    def colors(v,keep):
        import colorsys
        hue=(np.arctan2(v[:,:,1],v[:,:,0])/(2*np.pi))%1
        palette=np.array([colorsys.hsv_to_rgb(i/360,1,1) for i in range(360)])
        rgb=(palette[(hue*360).astype(int)%360]*255).astype('u1');rgb[~keep]=0
        return Image.fromarray(rgb).resize((768,230),Image.Resampling.NEAREST)
    font=ImageFont.load_default(size=20);small=ImageFont.load_default(size=17)
    proc=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s','1536x680','-framerate',str(1/dt),'-i','pipe:0','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'dense_direction_maps.mp4')],stdin=subprocess.PIPE)
    try:
        for i in range(n):
            end,ek=field(arrays[0][i]);rates=arrays[1][i];av=rates/(1+rates.sum(axis=0));avg,ak=field(av)
            ref=np.zeros_like(end);rk=np.zeros((h,w),bool)
            if i:ref,rk,_=reference(np.asarray(images[i-1]),np.asarray(images[i]))
            common=ek&ak&rk
            errors=[]
            for v in [end,avg]:
                cos=(v*ref).sum(axis=2)/np.maximum(np.linalg.norm(v,axis=2)*np.linalg.norm(ref,axis=2),1e-12)
                errors.append(np.degrees(np.arccos(np.clip(cos,-1,1))))
            for key,(x0,y0,x1,y1) in regions.items():
                mask=common[y0:y1,x0:x1];row=dict(frame=i,region=key,common_pixels=int(mask.sum()),warmup=i<20)
                for j,label in enumerate(['end','frame_average']):
                    vals=errors[j][y0:y1,x0:x1][mask];row[label+'_median_deg']=describe(vals)['median_deg']
                    if i>=20:allerr[key][j].extend(vals.tolist())
                rows.append(row)
            canvas=Image.new('RGB',(1536,680),(15,20,30));d=ImageDraw.Draw(canvas)
            d.text((12,5),f'KITTI04 frame {i}  t={i*dt:.3f}s   Every model pixel is shown; black = excluded/uncertain, NOT static',fill='white',font=font)
            for label,x,y,im in [('Input',0,40,Image.fromarray(np.asarray(images[i])).convert('RGB').resize((768,230),Image.Resampling.NEAREST)),
                ('MT frame-end 20ms trace (native MSTd input)',768,40,colors(end,ek)),
                ('MT 104ms observed spike-count average (diagnostic only)',0,335,colors(avg,ak)),
                ('Adjacent-image Farneback reference (NOT ground truth)',768,335,colors(ref,rk))]:
                d.text((x+10,y),label,fill='white',font=small);canvas.paste(im,(x,y+28))
            d=ImageDraw.Draw(canvas)
            import colorsys
            labels=['Right','Down-right','Down','Down-left','Left','Up-left','Up','Up-right']
            for j,label in enumerate(labels):
                col=tuple(int(v*255) for v in colorsys.hsv_to_rgb(j/8,1,1));x=12+j*190
                d.rectangle((x,613,x+20,633),fill=col);d.text((x+26,613),label,fill='white',font=small)
            d.text((12,650),'Same-frame comparison; V1/neural temporal filtering retained. No model change, lag fitting, or MSTd rerun.',fill='white',font=small)
            if i in [25,49]:canvas.save(out/f'frame_{i:06d}.png')
            proc.stdin.write(canvas.tobytes())
    finally:proc.stdin.close();code=proc.wait()
    assert code==0
    summary={k:dict(frame_end=describe(vals[0]),frame_average=describe(vals[1])) for k,vals in allerr.items()}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    with (out/'per_frame.csv').open('w',encoding='utf-8-sig',newline='') as f:
        cw=csv.DictWriter(f,fieldnames=list(rows[0]));cw.writeheader();cw.writerows(rows)
    cfg=dict(frames=n,frame_interval_s=dt,display='all160x48 model pixels in each direction map, no arrow subsampling',
        colors='HSV direction, clockwise from image right; fixed saturation/value; excluded pixels black',
        statistics='common reference AND frame-end AND frame-average masks; pooled pixel-frame samples after20-frame warmup',
        frame_average='observed MT excitatory counts /104ms, then same per-pixel normalization; no spike timing reconstruction',
        limitations='Reference is not GT; temporal kernels differ; no proof of improved MSTd because it was not rerun.')
    (out/'video_config.json').write_text(json.dumps(cfg,indent=2))
    artifact=dict(schema='mstd-artifact-v1',id=run.name+'/'+out.name,kind='visualization',status='completed',stimulus_ids=art['stimulus_ids'],
        input_paths=art['input_paths'],input_sha256=r['input']['stimulus_sha256'],parent_run_ids=[run.name],notes='Dense directions and offline MT readout-window diagnostic; no model change.')
    (out/'artifact.json').write_text(json.dumps(artifact,indent=2))
    print(json.dumps(summary))
if __name__=='__main__':main()
