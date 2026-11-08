"""Registered natural-stimulus MT runs, following docs/FILE_ORGANIZATION.md.

No resampling, renamed stimuli, or model changes. --dry-run validates the exact
indexed DAT and metadata, then prints paths/config without creating artifacts.
"""
import argparse
import csv
from datetime import datetime,timedelta,timezone
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mstd.calibration import inspect_stimulus,file_sha256
from mstd.model import load_model_config,model_config
from mstd.execution import execution_settings

DEFAULTS={
    'daria_walk':('daria_walk','daria_walk__180x144__f000084__dt50ms__gray_native__v01'),
    'paragliding':('davis_paragliding','davis_paragliding__854x480__f000070__dt50ms__gray_native__v01'),
    'snowboard':('snowboard','snowboard__854x480__f000065__dt50ms__gray_native__v01'),
}
FIELDS=['id','kind','path','legacy_path','status','parent_run_ids','stimulus_ids','notes']


def indexed_stimulus(root,sequence,stimulus_id=None):
    dataset,default=DEFAULTS[sequence];sid=stimulus_id or default
    with (root/'input/registry.csv').open(encoding='utf-8-sig',newline='') as stream:
        rows=[r for r in csv.DictReader(stream) if Path(r['path']).stem==sid]
    if len(rows)!=1:raise ValueError(f'expected exactly one indexed stimulus: {sid}')
    row=rows[0];path=(root/row['path']).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():raise FileNotFoundError(f'upload the indexed stimulus bundle: {row["path"]}')
    if not Path(row['path']).is_relative_to(Path('input/stimuli/natural')/dataset):raise ValueError('stimulus belongs to another dataset')
    info=inspect_stimulus(path);digest=file_sha256(path)
    if digest!=row['sha256'] or any(info[k]!=int(row[k]) for k in ('width','height','frames','channels')):
        raise ValueError('DAT disagrees with input registry')
    meta=json.loads(path.with_suffix('.dat.json').read_text(encoding='utf-8'))
    if meta.get('stimulus_sha256')!=digest or not isinstance(meta.get('frame_ms'),(int,float)) or meta['frame_ms']<=0:
        raise ValueError('matching stimulus metadata with explicit frame_ms required')
    return dataset,sid,row['path'],info,meta['frame_ms'],digest


def registry_update(root,row):
    """Short exclusive lock; update only this ID, preserving other registry rows."""
    path=root/'experiments/registry.csv';path.parent.mkdir(parents=True,exist_ok=True)
    lock=path.with_suffix('.lock');temp=path.with_suffix('.pending')
    with lock.open('x') as handle:
        try:
            records=[]
            if path.exists():
                with path.open(encoding='utf-8-sig',newline='') as stream:
                    reader=csv.DictReader(stream)
                    if reader.fieldnames!=FIELDS:raise ValueError('unexpected experiments registry schema')
                    records=list(reader)
            matches=[i for i,r in enumerate(records) if r['id']==row['id']]
            if len(matches)>1:raise ValueError('duplicate experiment ID in registry')
            if matches:
                previous=records[matches[0]]
                if row['status']=='planned' or previous['path']!=row['path'] or previous['kind']!=row['kind'] or previous['status'] not in ('planned','running'):
                    raise ValueError('existing artifact cannot be overwritten; choose another repeat/time')
                records[matches[0]]=row
            else:records.append(row)
            with temp.open('x',encoding='utf-8',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=FIELDS);writer.writeheader();writer.writerows(records)
            os.replace(temp,path)
        finally:
            handle.close()
            lock.unlink()


def plan(root,sequence,stage='recording',repeat=1,backend=None,device=None,max_frames=None,stimulus_id=None,view_frame=30,stamp=None):
    dataset,sid,path,info,frame_ms,digest=indexed_stimulus(root,sequence,stimulus_id)
    if type(repeat) is not int or repeat<1:raise ValueError('positive repeat required')
    if max_frames is not None and (type(max_frames) is not int or max_frames<1):raise ValueError('positive max_frames required')
    if view_frame<0:raise ValueError('nonnegative view frame required')
    execution=execution_settings(backend,device);backend,device=execution['backend'],execution['device']
    count=min(info['frames'],max_frames) if max_frames else info['frames']
    stamp=stamp or datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    tag='py_spike_'+('cuda' if device.startswith('cuda') else 'cpu')
    run=f'{stamp}__{dataset}_{info["width"]}x{info["height"]}_f{count:06d}__{tag}__r{repeat:02d}'
    recording=(Path('outputs/recordings')/dataset/run if stage=='recording' else Path('experiments/scratch')/run)
    view=Path('outputs/visualizations')/dataset/run/'mt_spikes__v01'
    for candidate in (recording,view):
        if (root/candidate).exists():raise FileExistsError(candidate)
    occupied=set()
    if (root/'experiments/registry.csv').exists():
        with (root/'experiments/registry.csv').open(encoding='utf-8-sig',newline='') as stream:
            occupied={r['id'] for r in csv.DictReader(stream)}
    if run in occupied or run+'__mt_spikes__v01' in occupied:raise ValueError('run ID already registered')
    config=load_model_config(root/'configs/model_python_spiking.json')
    config.update(input=dict(kind='stimulus_dat',path=path),output=recording.as_posix(),
                  backend=backend,device=device,frame_ms=frame_ms,max_frames=max_frames,progress_every=1)
    config['monitors'].update(spikes=True,neural_state=True,every_n_frames=1)
    config=model_config(config)
    return dict(run_id=run,dataset=dataset,stimulus_id=sid,input_path=path,input_sha256=digest,
                frames=count,input_frames=info['frames'],recording=recording.as_posix(),visualization=view.as_posix(),
                frame=min(view_frame,count-1),config=config,stage=stage)


def record(root,p,kind,status,error=''):
    is_view=kind=='visualization'
    identifier=p['run_id']+('__mt_spikes__v01' if is_view else '')
    path=p['visualization'] if is_view else p['recording']
    parents=[p['run_id']] if is_view else []
    note=f'python_spiking; frame_ms={p["config"]["frame_ms"]}; frames={p["frames"]}; seed={p["config"]["spiking"]["seed"]}'
    if error:note+='; '+error
    registry_update(root,dict(id=identifier,kind=kind,path=path,legacy_path='',status=status,
                             parent_run_ids=';'.join(parents),stimulus_ids=p['stimulus_id'],notes=note))
    if (root/path).exists():
        artifact=dict(schema='mstd-artifact-v1',id=identifier,kind=kind,status=status,
                      stimulus_ids=[p['stimulus_id']],input_paths=[p['input_path']],input_sha256=p['input_sha256'],
                      parent_run_ids=parents,notes=note,processed_frames=p['frames'],
                      source_sha256={name:file_sha256(root/name) for name in (
                          'tools/run_natural_spiking.py','src/mstd/v1.py','src/mstd/neural.py',
                          'src/mstd/spiking_mt.py','src/mstd/model.py','src/mstd/tensor_pipeline.py','src/mstd/execution.py',
                          'src/mstd/simulation.py','src/mstd/monitors.py','configs/v1_projection.json',
                          'examples/visualize_python_spiking.py','examples/render_model_mt.py') if (root/name).is_file()},
                      command=[sys.executable,*sys.argv],view_parameters=dict(frame=p['frame'],rate_max=300,direction_deg=0,video=True) if is_view else None)
        (root/path/'artifact.json').write_text(json.dumps(artifact,indent=2,ensure_ascii=False),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sequence',choices=list(DEFAULTS),required=True)
    parser.add_argument('--stimulus-id',help='explicit indexed alternative, e.g. snowboard 66-frame variant')
    parser.add_argument('--stage',choices=['scratch','recording'],default='scratch')
    parser.add_argument('--repeat',type=int,default=1)
    parser.add_argument('--max-frames',type=int,help='omit to process ALL input frames')
    parser.add_argument('--backend',choices=['numpy','torch'])
    parser.add_argument('--device',choices=['cpu','cuda','gpu'],help='default CUDA; cpu selects NumPy unless backend is explicit')
    parser.add_argument('--view-frame',type=int,default=30)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args();os.chdir(ROOT)
    p=plan(ROOT,args.sequence,args.stage,args.repeat,args.backend,args.device,args.max_frames,args.stimulus_id,args.view_frame)
    print(json.dumps(p,indent=2,ensure_ascii=False),flush=True)
    if args.dry_run:return
    if not shutil.which('ffmpeg'):raise RuntimeError('ffmpeg required for requested video')
    __import__('matplotlib')
    from mstd.simulation import simulate
    kind='recording' if args.stage=='recording' else 'scratch'
    record(ROOT,p,kind,'planned');record(ROOT,p,kind,'running')
    try:
        simulate(p['config'])
        report=json.loads((ROOT/p['recording']/'model_report.json').read_text(encoding='utf-8'))
        if report.get('complete') is not True or report['frames']!=p['frames'] or report['input']['input_sha256']!=p['input_sha256']:
            raise ValueError('model completion/frame/hash verification failed')
        record(ROOT,p,kind,'completed')
    except BaseException as error:
        (ROOT/p['recording']).mkdir(parents=True,exist_ok=True)
        record(ROOT,p,kind,'failed',str(error));raise
    record(ROOT,p,'visualization','planned');record(ROOT,p,'visualization','running')
    try:
        subprocess.run([sys.executable,'examples/visualize_python_spiking.py',p['recording'],
                        '--frame',str(p['frame']),'--rate-max','300','--direction-deg','0',
                        '--video','--stimulus',p['input_path'],'--output',p['visualization']],check=True)
        record(ROOT,p,'visualization','completed')
    except BaseException as error:
        (ROOT/p['visualization']).mkdir(parents=True,exist_ok=True)
        record(ROOT,p,'visualization','failed',str(error));raise
    print(f'Completed: {p["recording"]}\nVideo: {p["visualization"]}/mt_spiking.mp4')


if __name__=='__main__':main()
