"""Overlay recorded MT directions and MSTd/GT translation bearings, no model rerun.

Fixed arrow length encodes direction only: neither pixels/second nor object3D
velocity. No semantic object detection or instance tracking is performed.
"""
import argparse,csv,hashlib,json,math
from pathlib import Path
import shutil,subprocess,sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont


def arrow(draw,x,y,dx,dy,color,length=22):
    norm=math.hypot(dx,dy)
    if norm<1e-12:return
    ux,uy=dx/norm,dy/norm;ex,ey=x+length*ux,y+length*uy
    draw.line((x,y,ex,ey),fill=color,width=2)
    draw.line((ex,ey,ex-6*ux+3*uy,ey-6*uy-3*ux),fill=color,width=2)
    draw.line((ex,ey,ex-6*ux-3*uy,ey-6*uy+3*ux),fill=color,width=2)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording',required=True)
    p.add_argument('--project-root',default=str(Path(__file__).resolve().parents[1]))
    p.add_argument('--output');p.add_argument('--stimulus')
    p.add_argument('--first-frame',type=int,default=0);p.add_argument('--max-frames',type=int)
    p.add_argument('--stride',type=int,default=8)
    p.add_argument('--min-strength',type=float,default=.05)
    p.add_argument('--min-resultant-ratio',type=float,default=.2)
    p.add_argument('--no-register',action='store_true')
    a=p.parse_args();root=Path(a.project_root).resolve();sys.path.insert(0,str(root/'src'))
    from mstd.project_paths import resolve_stimulus
    from mstd.calibration import inspect_stimulus,file_sha256
    if a.stride<1 or not 0<=a.min_strength<=1 or not 0<=a.min_resultant_ratio<=1:p.error('invalid arrow settings')
    run=(root/a.recording).resolve()
    report=json.loads((run/'model_report.json').read_text(encoding='utf-8'))
    assert report['complete'] and report['model']['mt_backend']=='python_spiking'
    stimulus=resolve_stimulus(a.stimulus or report['input']['input_path'],report['input']['stimulus_sha256'],root)
    assert file_sha256(stimulus)==report['input']['stimulus_sha256']
    info=inspect_stimulus(stimulus);n=report['frames'];w=info['width'];h=info['height'];dt=report['model']['frame_interval_s']
    assert info['channels']==1 and report['model']['width']==w and report['model']['height']==h
    directory=run/'monitors/mt_readout_activity'
    meta=json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
    assert meta['complete'] and meta['shape']==[n,8,h,w] and meta['layout']=='TDYX' and meta['units']=='dimensionless'
    assert meta['sample_indices']==list(range(n)) and np.allclose(meta['timestamps_s'],(np.arange(n)+1)*dt,atol=1e-8)
    assert (directory/'values.f32').stat().st_size==n*8*h*w*4
    activity=np.memmap(directory/'values.f32',mode='r',dtype='<f4',shape=(n,8,h,w))
    pixels=np.memmap(stimulus,mode='r',dtype='u1',offset=25,shape=(info['frames'],h,w))
    estimates=[json.loads(s) for s in (run/'estimates.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(estimates)==n
    truth=None;sidecar=stimulus.with_suffix('.dat.json')
    if sidecar.exists():
        data=json.loads(sidecar.read_text(encoding='utf-8'))
        if data.get('truth_file'):
            truth=[json.loads(s) for s in (stimulus.parent/data['truth_file']).read_text(encoding='utf-8').splitlines()]
            assert len(truth)>=n
    if truth is None and (stimulus.parent/(stimulus.stem+'.truth.jsonl')).is_file():
        truth=[json.loads(s) for s in (stimulus.parent/(stimulus.stem+'.truth.jsonl')).read_text(encoding='utf-8').splitlines()]
    start=a.first_frame;end=min(n,start+a.max_frames) if a.max_frames else n
    if not 0<=start<end<=n:p.error('empty/invalid frame range')
    out=Path(a.output).resolve() if a.output else root/'outputs/visualizations/kitti'/run.name/'mt_heading_overlay__v01'
    if not a.no_register and not out.is_relative_to(root/'outputs/visualizations'):p.error('use outputs/visualizations or --no-register')
    encoder=shutil.which('ffmpeg');probe=shutil.which('ffprobe')
    if not encoder or not probe:p.error('ffmpeg and ffprobe required')
    out.mkdir(parents=True,exist_ok=False)
    width=1280;image_height=2*round(h*width/w/2);height=image_height+144;scale_x=width/w;scale_y=image_height/h
    movie=out/'mt_heading_overlay.mp4'
    process=subprocess.Popen([encoder,'-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','rgb24',
        '-s',f'{width}x{height}','-framerate',str(1/dt),'-i','pipe:0','-an','-c:v','libx264','-crf','18',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(movie)],stdin=subprocess.PIPE)
    angles=np.array(meta['directions_rad']);intrinsics=report['model']['intrinsics_px']
    try:font=ImageFont.load_default(size=18)
    except TypeError:font=ImageFont.load_default()
    try:
        for i in range(start,end):
            s=estimates[i];assert s['frame']==i and abs(s['estimate_timestamp_s']-(i+1)*dt)<1e-8
            value=np.asarray(activity[i]);assert np.isfinite(value).all() and (value>=0).all()
            vx=np.einsum('dyx,d->yx',value,np.cos(angles));vy=np.einsum('dyx,d->yx',value,np.sin(angles))
            mag=np.hypot(vx,vy);total=value.sum(axis=0);ratio=mag/np.maximum(total,1e-12)
            canvas=Image.new('RGB',(width,height),(12,16,24))
            canvas.paste(Image.fromarray(np.asarray(pixels[i])).resize((width,image_height),Image.Resampling.NEAREST).convert('RGB'),(0,96))
            draw=ImageDraw.Draw(canvas)
            for y in range(a.stride//2,h,a.stride):
                for x in range(a.stride//2,w,a.stride):
                    if mag[y,x]>=a.min_strength and ratio[y,x]>=a.min_resultant_ratio:
                        arrow(draw,(x+.5)*scale_x,96+(y+.5)*scale_y,float(vx[y,x]),float(vy[y,x]),(0,255,230))
            gt=None;vertical=0.
            if truth is not None:
                t=truth[i];assert t['frame']==i
                gt=t.get('expected_heading_deg',t.get('camera',{}).get('heading_camera_deg'))
                vertical=t.get('vertical_bearing_deg',0.)
            predicted=s['alpha_x_deg'] if s['bearing_valid'] else None
            error=abs((predicted-gt+180)%360-180) if predicted is not None and gt is not None else None
            fmt=lambda v:'--' if v is None else f'{v:.3f}'
            winner=s.get('winner_alpha_x_deg')
            if winner is None and intrinsics and s['bearing_valid']:
                cx=report['model']['mstd_template_centers_px'][s['winner_index']][0]
                winner=math.degrees(math.atan((cx-intrinsics['cx_px'])/intrinsics['fx_px']))
            draw.text((16,10),f'Frame {i}  input {i*dt:.3f}s  estimate available {(i+1)*dt:.3f}s  warmup={i<20}',fill='white',font=font)
            draw.text((16,34),f'GT bearing={fmt(gt)} deg   MSTd={fmt(predicted)} deg   valid={s["bearing_valid"]}   error={fmt(error)} deg',fill='white',font=font)
            draw.text((16,58),f'Winner={fmt(winner)}   local quadratic={fmt(s.get("local_quadratic_alpha_x_deg"))}   cyan: MT image directions',fill='white',font=font)
            if intrinsics:
                for alpha,beta,color,label in ((gt,vertical,(255,210,0),'GT translation FoE'),(predicted,s.get('alpha_y_deg') or 0.,(255,70,100),'MSTd FoE')):
                    if alpha is None:continue
                    x=(intrinsics['cx_px']+intrinsics['fx_px']*math.tan(math.radians(alpha))+.5)*scale_x
                    y=96+(intrinsics['cy_px']+intrinsics['fy_px']*math.tan(math.radians(beta))+.5)*scale_y
                    if 0<=x<width and 96<=y<height-48:
                        draw.ellipse((x-8,y-8,x+8,y+8),outline=color,width=2)
                        draw.text((x+10,y-25 if label.startswith('GT') else y+12),label,fill=color,font=font)
            draw.text((16,height-34),'Fixed arrows: direction only, NOT px/s or object3D speed. No object recognition. Filter latency retained.',fill='white',font=font)
            if i==start:canvas.save(out/'first_frame.png')
            process.stdin.write(canvas.tobytes())
    finally:
        process.stdin.close();code=process.wait();activity._mmap.close();pixels._mmap.close()
    if code:raise RuntimeError('ffmpeg failed')
    video=json.loads(subprocess.check_output([probe,'-v','error','-select_streams','v:0','-show_entries','stream=codec_name,nb_frames,avg_frame_rate,duration','-of','json',str(movie)],text=True))['streams'][0]
    assert video['codec_name']=='h264' and int(video['nb_frames'])==end-start
    settings=dict(source_run=run.name,first_frame=start,end_frame_exclusive=end,frame_interval_s=dt,
        overlay='fixed-length MT normalized-readout direction arrows; not metric optical flow',stride=a.stride,
        min_strength=a.min_strength,min_resultant_ratio=a.min_resultant_ratio,objects='not segmented/tracked',
        timing='frame-end MT paired to its processed input frame; filter latency not removed',video=video)
    (out/'video_config.json').write_text(json.dumps(settings,indent=2),encoding='utf-8')
    artifact=dict(schema='mstd-artifact-v1',id=run.name+'/'+out.name,kind='visualization',status='completed',stimulus_ids=[],
        input_paths=[str(stimulus)],input_sha256=report['input']['stimulus_sha256'],parent_run_ids=[run.name],notes=settings['overlay'])
    (out/'artifact.json').write_text(json.dumps(artifact,indent=2),encoding='utf-8')
    if not a.no_register:
        registry=root/'experiments/registry.csv'
        with registry.open(encoding='utf-8-sig',newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;assert not any(r['id']==artifact['id'] for r in reader)
        with registry.open('a',encoding='utf-8',newline='') as f:csv.DictWriter(f,fieldnames=fields).writerow(dict(id=artifact['id'],kind='visualization',path=out.relative_to(root).as_posix(),status='completed',parent_run_ids=run.name,notes=artifact['notes']))
    print(json.dumps(dict(video=str(movie),frames=end-start,codec='h264')))


if __name__=='__main__':main()
