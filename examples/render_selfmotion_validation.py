"""Validate and render one saved independent spiking self-motion run; no model changes."""
import argparse,csv,json,math,shutil,subprocess
from pathlib import Path
import numpy as np
from visualize_python_spiking import load_recording,close_recording,paired_stimulus,RATE_NAMES
from render_model_mt import Renderer,WIDTH,HEIGHT
import visualize_mt_gratings as draw

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording',required=True);p.add_argument('--output',required=True)
    p.add_argument('--rate-max',type=float,default=300.)
    a=p.parse_args();run=Path(a.recording);out=Path(a.output)
    if not math.isfinite(a.rate_max) or a.rate_max<=0:p.error('positive rate-max required')
    data=load_recording(run)
    process=None
    try:
        stimulus,info=paired_stimulus(data)
        report=data['report'];n=report['frames'];dt=report['model']['frame_interval_s']
        assert data['indices']==list(range(n))
        manifest=json.loads((stimulus.parent/'manifest.json').read_text(encoding='utf-8'))
        case=next(r for r in manifest['runs'] if r['stimulus']==stimulus.name)
        static=case['motion']['mode']=='static'
        if not static and case['motion']['mode']!='straight':raise ValueError('only static/straight baseline supported')
        expected=None if static else case['motion']['heading_deg']
        assert report['model']['intrinsics_px']==manifest['intrinsics_px']
        estimates=[json.loads(line) for line in (run/'estimates.jsonl').open(encoding='utf-8')]
        assert len(estimates)==n
        for i,e in enumerate(estimates):
            assert e['frame']==i and abs(e['estimate_timestamp_s']-(i+1)*dt)<1e-8
        candidates=np.asarray([math.degrees(math.atan((c[0]-report['model']['intrinsics_px']['cx_px'])/report['model']['intrinsics_px']['fx_px'])) for c in report['model']['mstd_template_centers_px']])
        for layer,key in [('mstd_scores','template_scores'),('mstd_activity','activity')]:
            directory=run/'monitors'/layer;m=json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
            assert m['complete'] and m['shape']==[n,len(candidates)] and m['sample_indices']==list(range(n))
            assert (directory/'values.f32').stat().st_size==n*len(candidates)*4
            v=np.fromfile(directory/'values.f32',dtype='<f4').reshape(n,-1)
            assert np.isfinite(v).all() and np.allclose(v,np.array([e[key] for e in estimates]),atol=1e-7,rtol=0)
        out.mkdir(parents=True,exist_ok=False)
        renderer=Renderer(info['width'],info['height'],a.rate_max,1.,'jet',
            titles=('MT EXCITATORY SPIKE RATE - HZ','MT INHIBITORY SPIKE RATE - HZ','MT READOUT ADAPTER - NOT SPIKES'),
            subtitle='SAVED SPIKING MT AND CONTINUOUS MSTD - NO RECOMPUTATION')
        full_height=HEIGHT+380
        encoder=shutil.which('ffmpeg')
        if not encoder:raise RuntimeError('ffmpeg is required')
        command=[encoder,'-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s',f'{WIDTH}x{full_height}',
            '-r',str(1/dt),'-i','-','-an','-c:v','libx264','-preset','veryfast','-crf','18',
            '-pix_fmt','yuv420p','-movflags','+faststart',str(out/'selfmotion_response.mp4')]
        process=subprocess.Popen(command,stdin=subprocess.PIPE)
        frames=[]
        with stimulus.open('rb') as source:
            source.seek(25)
            for i,e in enumerate(estimates):
                pixels=np.frombuffer(source.read(info['width']*info['height']),dtype='u1').reshape(info['height'],info['width'])
                stages=[data['saved'][name][1][i][data['order']] for name in RATE_NAMES]
                top=renderer.render(pixels,stages,i,n,dt)
                top[132:210,285:1435]=(15,21,33)
                renderer.text(top,295,141,'INPUT: '+case['name'].replace('_',' '),3,(255,205,70))
                renderer.text(top,295,183,'TRUE HEADING: UNDEFINED - STATIONARY' if static else f'TRUE TRANSLATION HEADING: {expected:g} DEG')
                bottom=np.empty((380,WIDTH,3),dtype='u1');bottom[:]=(15,21,33)
                # Render on a standard-height canvas so the shared font renderer remains in bounds.
                chart=np.empty((HEIGHT,WIDTH,3),dtype='u1');chart[:]=(15,21,33)
                renderer.text(chart,20,15,'MSTD: CYAN = MATCH SCORE   YELLOW = ACTIVITY   FIXED SCALE 0 TO 1',2)
                heading=e['alpha_x_deg'];label='INVALID' if not e['bearing_valid'] else f'{heading:.2f} DEG'
                renderer.text(chart,20,48,f'OUTPUT: {label}  MARGIN {e["margin"]:.5f}  LOCAL EVIDENCE {e["mt_evidence"]:.3f}',2)
                for j,(score,activity,angle) in enumerate(zip(e['template_scores'],e['activity'],candidates)):
                    x=70+j*150
                    chart[280-int(max(0,min(1,score))*175):280,x:x+35]=(60,225,245)
                    chart[280-int(max(0,min(1,activity))*175):280,x+40:x+75]=(255,205,70)
                    renderer.text(chart,x-8,296,f'{angle:.0f} DEG',2)
                    if j==e['winner_index']:renderer.text(chart,x-10,326,'WINNER',1,(255,205,70))
                if static:renderer.text(chart,20,355,'STATIC: ANY VALID HEADING IS A FALSE POSITIVE',2)
                else:renderer.text(chart,20,355,'CHECK BOTH HEADING ERROR AND VALID COVERAGE',2)
                bottom[:]=chart[:380]
                canvas=np.concatenate([top,bottom])
                process.stdin.write(canvas.tobytes())
                if i in (0,20,40,n-1):draw.png(out/f'frame_{i+1:06d}.png',canvas.tobytes(),WIDTH,full_height)
                frames.append(dict(frame=i,end_s=e['estimate_timestamp_s'],bearing_valid=e['bearing_valid'],
                    heading=heading,unconditional_winner_deg=float(candidates[e['winner_index']]),
                    true_heading=expected,margin=e['margin'],max_score=max(e['template_scores']),
                    max_activity=max(e['activity']),evidence=e['mt_evidence']))
        process.stdin.close()
        if process.wait()!=0:raise RuntimeError('ffmpeg failed')
        stable=frames[20:];valid=[f for f in stable if f['bearing_valid']]
        errors=[abs((f['heading']-expected+180)%360-180) for f in valid] if not static else []
        metrics=dict(case=case['name'],frames=n,excluded_initial_frames=20,stable_frames=len(stable),
            valid_frames=len(valid),valid_fraction=len(valid)/len(stable),
            static_false_valid_fraction=len(valid)/len(stable) if static else None,
            heading_mae_valid_frames=float(np.mean(errors)) if errors else None,
            heading_p95_valid_frames=float(np.percentile(errors,95)) if errors else None,
            winner_angles_even_when_invalid=sorted(set(f['unconditional_winner_deg'] for f in stable)),
            mean_max_score=float(np.mean([f['max_score'] for f in stable])),
            mean_max_activity=float(np.mean([f['max_activity'] for f in stable])),
            input_sha256=report['input']['stimulus_sha256'],
            checks='MT native loader, stimulus hash, frame timing, intrinsics, MSTd JSON/binary consistency validated',
            limitations='One seed and one scene; no task acceptance threshold fitted; invalid estimates excluded only from conditional error.')
        (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
        with (out/'heading.csv').open('x',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(frames[0]));writer.writeheader();writer.writerows(frames)
        (out/'video_config.json').write_text(json.dumps(dict(frames=n,fps=1/dt,rate_max=a.rate_max,
            direction_order=list(range(0,360,45)),mstd_scale=[0,1],truth=expected,
            recording=a.recording,stimulus=str(stimulus),units='MT E/I Hz; readout and MSTd dimensionless',
            response_timestamp='frame end',unconditional_winner_not_valid_estimate=True),indent=2),encoding='utf-8')
        artifact=dict(schema='mstd-artifact-v1',id=run.name+'/'+out.name,kind='visualization',status='completed',
            stimulus_ids=[stimulus.parent.name+'__'+case['name']],input_paths=[str(stimulus)],
            input_sha256=metrics['input_sha256'],parent_run_ids=[run.name],notes=metrics['limitations'])
        (out/'artifact.json').write_text(json.dumps(artifact,indent=2),encoding='utf-8')
        root=Path(__file__).resolve().parents[1];registry=root/'experiments/registry.csv'
        with registry.open(encoding='utf-8-sig',newline='') as f:fields=next(csv.reader(f))
        with registry.open('a',encoding='utf-8',newline='') as f:
            csv.DictWriter(f,fieldnames=fields).writerow(dict(id=artifact['id'],kind='visualization',path=out.as_posix(),
                status='completed',parent_run_ids=run.name,stimulus_ids=artifact['stimulus_ids'][0],notes=artifact['notes']))
        print(json.dumps(metrics,indent=2))
    finally:
        if process is not None and process.poll() is None:process.kill();process.wait()
        close_recording(data)

if __name__=='__main__':main()
