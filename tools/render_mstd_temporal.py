"""Shared-scale time curves of GT and accepted heading, written to MP4."""
import argparse,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'examples'))
import visualize_mt_gratings as draw
from mstd.v1_recording import dump_json
from mstd.calibration import file_sha256
from run_mt_spatial_controls import register_run


def line(canvas,x0,y0,x1,y1,color):
    steps=max(abs(x1-x0),abs(y1-y0),1)
    for k in range(steps+1):draw.rect(canvas,round(x0+(x1-x0)*k/steps),round(y0+(y1-y0)*k/steps),2,2,color)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--case',default='step');a=p.parse_args()
    run=a.run.resolve();plan=json.loads((run/'plan.json').read_text())
    if a.case not in plan['cases']:raise ValueError('case must belong to this run')
    encoder=shutil.which('ffmpeg')
    if not encoder:raise RuntimeError('ffmpeg required')
    series={name:[json.loads(x) for x in (run/a.case/(name+'.jsonl')).read_text().splitlines()] for name in plan['variants']}
    cid=run.name+'__'+a.case+'__curves__v02';out=ROOT/'outputs/comparisons/mstd_temporal'/cid;out.mkdir(parents=True,exist_ok=False)
    artifact=dict(schema='mstd-artifact-v1',id=cid,kind='comparison',status='running',parent_run_ids=[run.name],stimulus_ids=[],input_paths=[str(run/a.case)],
        source_hashes={'tools/render_mstd_temporal.py':file_sha256(Path(__file__))},notes='Gold ground truth, cyan accepted heading. Invalid outputs appear as gaps. Shared minus22to22degree scale.')
    command=[encoder,'-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s','1440x640','-r',str(1/plan['frame_s']),'-i','pipe:0','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(out/'heading_curves.mp4')]
    proc=None
    try:
        with (out/'ffmpeg.log').open('x') as log:
            proc=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=log)
            for frame in range(plan['frames']):
                canvas=bytearray(bytes((15,19,27))*draw.W*draw.H)
                draw.text(canvas,20,18,f'{a.case} FRAME {frame}  GOLD GT  CYAN ACCEPTED HEADING',2)
                for panel,(name,rows) in enumerate(series.items()):
                    x=25+(panel%3)*475;y=90+(panel//3)*265
                    draw.text(canvas,x,y-30,name,2);draw.text(canvas,x,y-10,'+22',1);draw.text(canvas,x,y+195,'-22 DEG',1)
                    draw.text(canvas,x+80,y+210,f'0 TO {plan["frames"]*plan["frame_s"]:g} S',1)
                    draw.rect(canvas,x,y+100,430,1,(70,75,85));draw.rect(canvas,x,y,1,200,(70,75,85))
                    for key,color in [('heading_deg',(255,205,70)),('alpha_x_deg',(70,225,235))]:
                        previous=None
                        for i in range(frame+1):
                            value=rows[i][key]
                            if value is None:previous=None;continue
                            point=(x+round(i*430/max(plan['frames']-1,1)),y+round((22-max(-22,min(22,value)))/44*200))
                            if previous:line(canvas,*previous,*point,color)
                            previous=point
                if frame==plan['frames']-1:draw.png(out/'last_frame.png',canvas)
                proc.stdin.write(canvas)
            proc.stdin.close()
            if proc.wait()!=0:raise RuntimeError('encoding failed')
        artifact['status']='completed'
        dump_json(out/'visualization.json',dict(frame_s=plan['frame_s'],frames=plan['frames'],ylim_deg=[-22,22],truth='gold',accepted_heading='cyan',invalid='gaps'))
    except BaseException:
        artifact['status']='failed'
        if proc and proc.poll() is None:proc.kill();proc.wait()
        raise
    finally:register_run(out,artifact)
    print(out)


if __name__=='__main__':main()
