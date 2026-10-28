"""Annotated MP4 previews of a single VisualStimulus DAT or every DAT in a folder.

Reads real image pixels, not neural responses; labels come from the matching manifest.
Requires only Python stdlib, mstd source, and ffmpeg. Output labels never enter DATs.
"""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
from time import perf_counter

from mstd.calibration import inspect_stimulus, file_sha256
import visualize_mt_gratings as view


def labels(path, explicit_manifest=None):
    manifest=Path(explicit_manifest) if explicit_manifest else path.parent/'manifest.json'
    if not manifest.exists():
        return None,[],None
    meta=json.loads(manifest.read_text(encoding='utf-8-sig'))
    if isinstance(meta.get('runs'),dict):
        run=meta['runs']
        if path.name != run['stimulus'] and explicit_manifest is None:
            raise ValueError('manifest stimulus filename mismatch')
        if file_sha256(path) != run['stimulus_sha256']:
            raise ValueError('stimulus does not match manifest checksum')
        with (manifest.parent/run['segments']).open(encoding='utf-8-sig',newline='') as stream:
            segments=list(csv.DictReader(stream))
        for s in segments:
            for k in ('first_frame','last_frame','start_ms','end_ms'):
                s[k]=int(s[k])
        return meta,segments,manifest
    run=next((r for r in meta.get('runs',[]) if r['stimulus']==path.name),None)
    if run is None:
        raise ValueError(f'{path}: no matching run in manifest')
    if file_sha256(path) != run['stimulus_sha256']:
        raise ValueError('source stimulus checksum mismatch')
    segment=dict(first_frame=1,last_frame=meta['frames'],start_ms=0,end_ms=meta['duration_ms'],
        name=run['name'],kind=meta.get('kind',run.get('motion',{}).get('mode','unknown')),
        expected_direction_deg=run.get('expected_direction_deg'),
        speed_px_per_frame=meta.get('speed_px_per_frame'),
        expected_heading_deg=run.get('motion',{}).get('heading_deg') if run.get('motion',{}).get('mode')=='straight' else None,
        phase='original_independent_case')
    return meta,[segment],manifest


def preview(path, output, ffmpeg, frame_ms=None, manifest=None, max_frames=None, playback_rate=1., progress_every=100):
    path,output=Path(path),Path(output)
    info=inspect_stimulus(path)
    if info['channels']!=1:
        raise ValueError('preview currently requires single-channel grayscale DAT')
    meta,segments,used_manifest=labels(path,manifest)
    ms=frame_ms if frame_ms is not None else meta['frame_ms'] if meta else None
    if ms is None or ms<=0:
        raise ValueError('DAT does not store frame timing: provide --frame-ms or a matching manifest')
    if meta and (meta['frame_ms']!=ms or (meta['width'],meta['height'])!=(info['width'],info['height'])):
        raise ValueError('frame timing or dimensions disagree with manifest')
    if segments:
        end=0
        for s in segments:
            if s['first_frame'] != end+1 or s['last_frame']<s['first_frame'] or s['start_ms']!=end*ms or s['end_ms']!=s['last_frame']*ms:
                raise ValueError('noncontiguous or misaligned segment table')
            end=s['last_frame']
        if end != info['frames']:
            raise ValueError('segment table does not cover DAT frames')
    total=min(info['frames'],max_frames or info['frames'])
    video=output/(path.stem+'.mp4')
    cmd=[ffmpeg,'-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','rgb24',
         '-s',f'{view.W}x{view.H}','-r',str(1000/ms*playback_rate),'-i','-',
         '-an','-c:v','libx264','-preset','veryfast','-crf','18','-pix_fmt','yuv420p',
         '-movflags','+faststart',str(video)]
    process=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    index=0
    started=perf_counter()
    print(f'[start] {path.name}: {total} frames at {1000/ms:g} fps',flush=True)
    try:
        with path.open('rb') as stream:
            stream.seek(25)
            for i in range(total):
                pixels=stream.read(info['width']*info['height'])
                if len(pixels)!=info['width']*info['height']:
                    raise ValueError('truncated frame')
                while segments and i+1>segments[index]['last_frame']:
                    index+=1
                s=segments[index] if segments else {}
                canvas=bytearray(bytes((15,21,33))*(view.W*view.H))
                view.text(canvas,20,20,'VISUAL INPUT STIMULUS - NOT A NEURAL RESPONSE',3)
                view.image(canvas,20,85,480,pixels,info['width'],info['height'])
                view.text(canvas,540,90,f'FRAME {i+1}/{total}  TIME {i*ms/1000:.2f} S',3)
                view.text(canvas,540,135,str(s.get('name',path.stem)).replace('_',' '),2)
                view.text(canvas,540,175,'PHASE: '+str(s.get('phase','unlabeled')).replace('_',' '),2)
                kind=str(s.get('kind','unknown'))
                view.text(canvas,540,215,'TYPE: '+kind.replace('_',' '),2)
                expected=s.get('expected_direction_deg')
                if expected not in (None,''):
                    direction=float(expected)%360
                    view.text(canvas,540,260,f'IMAGE MOTION: {view.NAMES[int(direction//45)%8]} {direction:g} DEG',2,(255,205,70))
                    view.arrow(canvas,650,380,direction,65)
                elif kind=='static':
                    view.text(canvas,540,260,'STATIONARY TEXTURE / SCENE - NO MOTION DIRECTION',2,(255,205,70))
                elif s.get('expected_heading_deg') not in (None,''):
                    heading=float(s['expected_heading_deg'])
                    view.text(canvas,540,260,f'CAMERA TRANSLATION HEADING: {heading:g} DEG',2,(255,205,70))
                    view.text(canvas,540,300,'LOCAL IMAGE MOTION VARIES ACROSS SPACE',2)
                speed=s.get('speed_px_per_frame')
                is_selfmotion = meta and (meta.get('suite_type')=='selfmotion' or meta.get('schema')=='selfmotion-stimulus-v1')
                if is_selfmotion:
                    entries=meta.get('source_runs',meta.get('runs',[]))
                    case=next((r for r in entries if r.get('name')==s.get('name')),None) if isinstance(entries,list) else None
                    motion=case.get('motion',{}) if case else {}
                    if motion.get('mode') in ('static','yaw'):
                        view.text(canvas,540,450,'CAMERA TRANSLATION SPEED: 0 M/S',2)
                    elif motion.get('speed_mps') is not None:
                        view.text(canvas,540,450,f'CAMERA SPEED: {motion["speed_mps"]:g} M/S',2)
                elif speed not in (None,''):
                    view.text(canvas,540,450,f'TEXTURE SPEED: {float(speed):g} PX/FRAME',2)
                view.text(canvas,540,500,'IMAGE X RIGHT / Y DOWN; IMAGE ANGLES CLOCKWISE',2)
                view.text(canvas,20,600,'ANNOTATIONS ARE PREVIEW ONLY - DAT PIXELS ARE UNMODIFIED',2)
                process.stdin.write(canvas)
                middle=(s.get('first_frame',1)+s.get('last_frame',total))//2
                if i==0 or i+1==middle:
                    view.png(output/f'{path.stem}_segment_{index+1:03d}.png',canvas)
                if progress_every and (i==0 or (i+1)%progress_every==0 or i+1==total):
                    elapsed=perf_counter()-started
                    print(f'[progress] {i+1}/{total}, elapsed={elapsed:.1f}s',flush=True)
        process.stdin.close()
        if process.wait()!=0:
            raise RuntimeError('ffmpeg failed')
    except BaseException:
        if process.poll() is None:
            process.kill()
        process.wait()
        raise
    result=dict(input=str(path.resolve()),input_sha256=file_sha256(path),frames=total,
        full_source=total==info['frames'],frame_ms=ms,playback_rate=playback_rate,
        video=video.name,manifest=str(used_manifest.resolve()) if used_manifest else None)
    (output/(path.stem+'.preview.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(f'[done] {video}',flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,help='DAT file or folder containing DATs')
    p.add_argument('--output',required=True,help='new output folder')
    p.add_argument('--manifest',help='explicit matching manifest; single DAT input only')
    p.add_argument('--frame-ms',type=int)
    p.add_argument('--max-frames',type=int)
    p.add_argument('--playback-rate',type=float,default=1.)
    p.add_argument('--progress-every',type=int,default=100)
    p.add_argument('--ffmpeg',default='ffmpeg')
    a=p.parse_args()
    if (a.max_frames is not None and a.max_frames<=0 or a.playback_rate<=0 or a.progress_every<0):
        p.error('invalid frame limit, playback or progress')
    ffmpeg=shutil.which(a.ffmpeg)
    if not ffmpeg:
        p.error('ffmpeg is required for MP4 export')
    source=Path(a.input)
    files=sorted(source.glob('*.dat')) if source.is_dir() else [source]
    if not files or source.is_dir() and a.manifest:
        p.error('no DAT files found, or --manifest used with a folder')
    output=Path(a.output)
    output.mkdir(parents=True,exist_ok=False)
    for path in files:
        preview(path,output,ffmpeg,a.frame_ms,a.manifest,a.max_frames,a.playback_rate,a.progress_every)


if __name__=='__main__':
    main()
