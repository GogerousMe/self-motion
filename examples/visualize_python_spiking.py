"""Plot Python spiking MT recordings; optional paired input/E/I/readout MP4.

Frame numbers are zero-based ORIGINAL input indices. Rates/voltages are saved
frame samples; raster times come from the actual per-tick spike count events.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import shutil
import subprocess

import numpy as np

from mstd.calibration import file_sha256,inspect_stimulus

RATE_NAMES=('mt_exc_rates','mt_inh_rates','mt_readout_activity')
EVENT_DTYPE=np.dtype([('tick','<u8'),('neuron','<u4'),('count','<u4')])


def load_recording(root):
    saved={}
    try:return _load_recording(root,saved)
    except BaseException:
        for _,values in saved.values():values._mmap.close()
        raise


def close_recording(data):
    for _,values in data['saved'].values():values._mmap.close()


def _load_recording(root,saved):
    root=Path(root)
    report=json.loads((root/'model_report.json').read_text(encoding='utf-8'))
    if report.get('complete') is not True or report['model'].get('mt_backend')!='python_spiking':
        raise ValueError('completed python_spiking model run required')
    model=report['model'];shape=(8,model['height'],model['width'])
    indices=None;directions=None
    for name in (*RATE_NAMES,'mt_filtered_rates','mt_exc_voltage','mt_inh_voltage'):
        path=root/'monitors'/name/'metadata.json'
        if not path.is_file():
            if name in RATE_NAMES:raise ValueError(f'missing {name}: enable MT monitors and rerun')
            continue
        meta=json.loads(path.read_text(encoding='utf-8'))
        ix=meta.get('sample_indices',[]);times=np.asarray(meta.get('timestamps_s',[]),float)
        if (not ix or any(type(i) is not int or not 0<=i<report['frames'] for i in ix)
                or any(a>=b for a,b in zip(ix,ix[1:]))
                or meta.get('schema')!='mstd-layer-monitor-v1' or meta.get('complete') is not True
                or meta.get('dtype')!='float32-le' or meta.get('layout')!='TDYX'
                or meta.get('file')!='values.f32' or meta.get('shape')!=[len(ix),*shape]
                or times.shape!=(len(ix),) or not np.isfinite(times).all()
                or not np.allclose(times,(np.asarray(ix)+1)*model['frame_interval_s'],atol=1e-9,rtol=0)):
            raise ValueError(f'invalid {name} layout/frame-end timing')
        unit='dimensionless' if name=='mt_readout_activity' else 'mV' if name.endswith('voltage') else 'Hz'
        if meta.get('units')!=unit:raise ValueError(f'invalid {name} units')
        if indices is not None and indices!=ix:raise ValueError('monitor indices disagree')
        indices=ix
        if 'directions_rad' in meta:
            angles=np.asarray(meta['directions_rad'],float)
            if angles.shape!=(8,) or not np.isfinite(angles).all():raise ValueError('invalid directions')
            if directions is not None and not np.allclose(angles,directions,atol=1e-12,rtol=0):raise ValueError('direction order mismatch')
            directions=angles
        binary=path.parent/'values.f32'
        if binary.stat().st_size!=len(ix)*math.prod(shape)*4:raise ValueError(f'{name} payload size mismatch')
        values=np.memmap(binary,mode='r',dtype='<f4',shape=(len(ix),*shape))
        saved[name]=(meta,values)
        for frame in values:
            if not np.isfinite(frame).all() or (unit!='mV' and (frame<0).any()):raise ValueError('invalid monitor values')
    if directions is None:raise ValueError('direction labels missing')
    degrees=np.degrees(directions)%360
    errors=abs((degrees[None,:]-np.arange(8)[:,None]*45+180)%360-180)
    order=errors.argmin(axis=1)
    if len(set(order))!=8 or errors[np.arange(8),order].max()>1e-5:raise ValueError('eight cardinal/diagonal directions required')
    return dict(root=root,report=report,saved=saved,indices=indices,order=order)


def paired_stimulus(data,path=None):
    report=data['report']
    if path is None:
        if report['input'].get('input_kind')!='stimulus_dat':raise ValueError('V1 replay requires --stimulus for paired video')
        path=report['input']['input_path']
    from mstd.project_paths import resolve_stimulus
    path=resolve_stimulus(path,report['input'].get('stimulus_sha256'));info=inspect_stimulus(path)
    if file_sha256(path)!=report['input'].get('stimulus_sha256'):raise ValueError('stimulus SHA256 mismatch')
    model=report['model']
    if (info['width'],info['height'])!=(model['width'],model['height']) or info['frames']<report['frames']:
        raise ValueError('stimulus grid/duration mismatch')
    return path,info


def raster_events(data,group,direction,start_s,duration_s,max_neurons=128):
    root=data['root']/'monitors/spikes'
    meta=json.loads((root/'metadata.json').read_text(encoding='utf-8'))
    expected_ticks=round(data['report']['frames']*data['report']['model']['frame_interval_s']*1000/meta['dt_ms'])
    shape=[8,data['report']['model']['height'],data['report']['model']['width']]
    if (meta.get('schema')!='mstd-python-spike-count-events-v1' or meta.get('complete') is not True
            or meta.get('record_bytes')!=16 or meta.get('shapes',{}).get(group)!=shape
            or meta.get('ticks')!=expected_ticks or meta['dt_ms']<=0):raise ValueError('invalid spike metadata')
    path=root/f'{group}.events';count=meta['records'][group]
    if path.stat().st_size!=count*16:raise ValueError('spike event payload size mismatch')
    rows=np.memmap(path,mode='r',dtype=EVENT_DTYPE,shape=(count,)) if count else np.empty(0,EVENT_DTYPE)
    previous=-1
    for first in range(0,count,65536):
        chunk=rows[first:first+65536]
        if (int(chunk['tick'][0])<previous or (np.diff(chunk['tick'].astype(np.int64))<0).any()
                or (chunk['tick']>=expected_ticks).any() or (chunk['neuron']>=math.prod(shape)).any()
                or (chunk['count']==0).any()):raise ValueError('invalid spike time/id/count')
        previous=int(chunk['tick'][-1])
    p=shape[1]*shape[2]
    chosen=np.linspace(0,p-1,min(p,max_neurons),dtype=int)+int(data['order'][direction])*p
    lookup=np.full(8*p,-1,int);lookup[chosen]=np.arange(len(chosen))
    # Bound the slice by tick before converting to floating seconds.
    lo=np.searchsorted(rows['tick'],max(0,start_s*1000/meta['dt_ms']-1),side='left')
    hi=np.searchsorted(rows['tick'],(start_s+duration_s)*1000/meta['dt_ms']-1,side='left')
    subset=rows[lo:hi];local=lookup[subset['neuron']]
    keep=local>=0;subset=subset[keep];local=local[keep]
    total=len(subset)
    if total>200000:
        selection=np.linspace(0,total-1,200000,dtype=int);subset=subset[selection];local=local[selection]
    return (subset['tick'].astype(float)+1)*meta['dt_ms']/1000,local,subset['count'],len(chosen),total


def plot(data,output,frame,border,rate_max,direction,start_s,duration_s):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if frame not in data['indices']:raise ValueError(f'input frame {frame} was not saved')
    sample=data['indices'].index(frame);order=data['order']
    h,w=data['report']['model']['height'],data['report']['model']['width']
    if border<0 or 2*border>=min(h,w):raise ValueError('ROI border leaves no pixels')
    roi=(slice(border,h-border),slice(border,w-border))
    times=data['saved']['mt_exc_rates'][0]['timestamps_s']
    profiles={}
    for name,(meta,values) in data['saved'].items():
        if name.endswith('voltage'):continue
        maximum=1 if name=='mt_readout_activity' else rate_max
        fig,axes=plt.subplots(2,4,figsize=(13,6),layout='constrained')
        for ax,channel,deg in zip(axes.flat,order,range(0,360,45)):
            artist=ax.imshow(values[sample,channel],origin='upper',cmap='viridis',vmin=0,vmax=maximum)
            ax.set_title(f'{deg} deg');ax.set_xticks([]);ax.set_yticks([])
        fig.colorbar(artist,ax=list(axes.flat),label=meta['units'])
        fig.suptitle(f'{name}; input index {frame}; available at {times[sample]:.3f}s (fixed scale)')
        fig.savefig(output/f'{name}.png',dpi=130);plt.close(fig)
        profiles[name]=np.asarray(values[:,:,roi[0],roi[1]]).mean(axis=(2,3),dtype=np.float64)[:,order]
    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True,layout='constrained')
    for ax,name,label in zip(axes,RATE_NAMES[:2],('MT excitatory','MT inhibitory')):
        for d in range(8):ax.plot(times,profiles[name][:,d],label=f'{d*45} deg')
        ax.set_ylabel(label+' mean (Hz)');ax.legend(ncol=4,fontsize=8);ax.grid(alpha=.2)
    axes[-1].set_xlabel('Frame-end time (s)');fig.suptitle(f'ROI excludes {border} pixels per edge')
    fig.savefig(output/'mean_rates.png',dpi=140);plt.close(fig)
    with (output/'mean_rates.csv').open('x',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(['input_index','frame_end_s']+[f'{name}_{d*45}_deg' for name in profiles for d in range(8)])
        for i,index in enumerate(data['indices']):writer.writerow([index,times[i]]+[profiles[name][i,d] for name in profiles for d in range(8)])
    if all(name in data['saved'] for name in ('mt_exc_voltage','mt_inh_voltage')):
        channel=int(order[direction]);x,y=w//2,h//2
        fig,ax=plt.subplots(figsize=(12,4),layout='constrained')
        for name in ('mt_exc_voltage','mt_inh_voltage'):
            ax.plot(times,data['saved'][name][1][:,channel,y,x],'.-',label=name)
        ax.set(xlabel='Frame-end time (s)',ylabel='mV',title=f'Frame-end voltage samples, NOT spike waveform; {direction*45} deg pixel ({x},{y})')
        ax.legend();fig.savefig(output/'voltage_samples.png',dpi=140);plt.close(fig)
    if (data['root']/'monitors/spikes/metadata.json').exists():
        fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True,layout='constrained')
        for ax,group in zip(axes,('exc','inh')):
            t,ids,counts,n,total=raster_events(data,group,direction,start_s,duration_s)
            ax.scatter(t,ids,s=2+np.minimum(counts,5),marker='|')
            ax.set(ylabel=f'{group}: sampled pixel index',ylim=(-1,n),title=f'{direction*45} deg; {n} fixed sampled neurons; {len(t)}/{total} nonzero event rows shown')
        axes[-1].set(xlabel='Actual tick-end spike time (s)',xlim=(start_s,start_s+duration_s))
        fig.savefig(output/'spike_raster.png',dpi=160);plt.close(fig)


def video(data,output,stimulus,rate_max):
    from render_model_mt import Renderer,WIDTH,HEIGHT
    path,info=paired_stimulus(data,stimulus)
    count=data['report']['frames'];dt=data['report']['model']['frame_interval_s']
    if data['indices']!=list(range(count)):raise ValueError('video requires every_n_frames=1')
    encoder=shutil.which('ffmpeg')
    if encoder is None:raise RuntimeError('MP4 requires ffmpeg on PATH')
    renderer=Renderer(info['width'],info['height'],rate_max,1,'jet',
                      titles=('MT EXCITATORY SPIKE RATE - HZ','MT INHIBITORY SPIKE RATE - HZ','RATE READOUT ADAPTER - NOT A NEURAL NORMALIZATION'),
                      subtitle='SAVED PYTHON NEURON SPIKES - FRAME BIN RATES')
    shape=(info['height'],info['width'])+((3,) if info['channels']==3 else ())
    size=math.prod(shape)
    command=[encoder,'-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s',f'{WIDTH}x{HEIGHT}',
             '-r',str(1/dt),'-i','pipe:0','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',
             '-movflags','+faststart',str(output/'mt_spiking.mp4')]
    with path.open('rb') as source,(output/'ffmpeg.log').open('x') as log:
        source.seek(25)
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=log)
        try:
            for i in range(count):
                pixels=np.frombuffer(source.read(size),dtype='u1').reshape(shape)
                stages=[data['saved'][name][1][i][data['order']] for name in RATE_NAMES]
                canvas=renderer.render(pixels,stages,i,count,dt)
                process.stdin.write(canvas.tobytes())
                if i in (0,count-1):
                    import matplotlib.pyplot as plt
                    plt.imsave(output/f'video_frame_{i:04d}.png',canvas)
            process.stdin.close()
            if process.wait()!=0:raise RuntimeError('ffmpeg failed; see ffmpeg.log')
        except BaseException:
            if process.poll() is None:process.kill()
            process.wait();raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording',help='one run folder containing model_report.json, not the suite parent')
    parser.add_argument('--frame',type=int,default=30)
    parser.add_argument('--output',required=True)
    parser.add_argument('--border',type=int,default=4)
    parser.add_argument('--rate-max',type=float,default=300)
    parser.add_argument('--direction-deg',type=int,choices=list(range(0,360,45)),default=0)
    parser.add_argument('--raster-start-s',type=float,default=0)
    parser.add_argument('--raster-duration-s',type=float,default=1)
    parser.add_argument('--video',action='store_true')
    parser.add_argument('--stimulus',help='original DAT, SHA256 checked; useful after moving recordings')
    args=parser.parse_args()
    if (not math.isfinite(args.rate_max) or args.rate_max<=0 or not math.isfinite(args.raster_start_s)
            or args.raster_start_s<0 or not math.isfinite(args.raster_duration_s) or args.raster_duration_s<=0):
        raise ValueError('invalid display scale/raster time window')
    data=load_recording(args.recording)
    try:
        if args.frame not in data['indices']:raise ValueError('selected frame was not recorded (zero-based index)')
        if args.video:paired_stimulus(data,args.stimulus)
        output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
        plot(data,output,args.frame,args.border,args.rate_max,args.direction_deg//45,args.raster_start_s,args.raster_duration_s)
        if args.video:video(data,output,args.stimulus,args.rate_max)
        (output/'visualization.json').write_text(json.dumps(dict(complete=True,recording=str(data['root'].resolve()),
            frame=args.frame,rate_max=args.rate_max,border=args.border,video=args.video,
            raster_direction_deg=args.direction_deg,raster_window_s=[args.raster_start_s,args.raster_start_s+args.raster_duration_s],
            note='frame maps/rates and frame-end voltages; raster uses actual tick events; fixed E/I scale'),indent=2),encoding='utf-8')
        print(output)
    finally:close_recording(data)


if __name__=='__main__':main()
