"""Frozen-image KITTI control: identical model parameters, no static heading GT."""
import argparse,csv,hashlib,json,sys
from datetime import datetime,timedelta,timezone
from pathlib import Path
import numpy as np

SID='kitti04_f000193_repeat__160x48__f000080__dt104ms__gray_static__v01'
ROOT=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def register(row):
    p=ROOT/'experiments/registry.csv'
    with p.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;known={r['id'] for r in reader}
    if row['id'] in known:raise FileExistsError(row['id'])
    with p.open('a',encoding='utf-8',newline='') as f:csv.DictWriter(f,fieldnames=fields).writerow(row)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan-only',action='store_true');a=p.parse_args()
    folder=ROOT/'input/stimuli/natural/kitti'/SID;dat=folder/(SID+'.dat');meta=load(dat.with_suffix('.dat.json'))
    sys.path.insert(0,str(ROOT/'src'));from mstd.calibration import inspect_stimulus
    from mstd.model import model_config
    assert inspect_stimulus(dat)['frames']==80 and sha(dat)==meta['stimulus_sha256']
    pixels=np.memmap(dat,mode='r',dtype='u1',offset=25,shape=(80,48,160));assert np.all(pixels==pixels[0])
    config=model_config(load(folder/'run_config.json'))
    assert config['frame_ms']==104 and config['spiking']['seed']==43
    if a.plan_only:
        print(json.dumps(dict(stimulus=dat.relative_to(ROOT).as_posix(),sha256=sha(dat),all80_frames_identical=True,frame_ms=104,
            spiking=config['spiking'],intrinsics=config['intrinsics'],monitors=config['monitors'],static_heading=None),indent=2));return
    from mstd.v1 import ArrayOps
    ArrayOps(config['backend'],config['device'])
    with (ROOT/'input/registry.csv').open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;known={r['path']:r for r in reader}
    rel=dat.relative_to(ROOT).as_posix()
    if rel not in known:
        with (ROOT/'input/registry.csv').open('a',encoding='utf-8',newline='') as f:
            csv.DictWriter(f,fieldnames=fields).writerow(dict(path=rel,width=160,height=48,frames=80,channels=1,sha256=sha(dat),scope='reusable'))
    else:assert known[rel]['sha256']==sha(dat)
    with (ROOT/'experiments/registry.csv').open(encoding='utf-8-sig',newline='') as f:known={r['id'] for r in csv.DictReader(f)}
    if SID not in known:register(dict(id=SID,kind='stimulus',path=folder.relative_to(ROOT).as_posix(),status='completed',stimulus_ids=SID,notes='Frozen middle frame of KITTI04 pilot; no defined translation heading.'))
    stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    rid=stamp+'__kitti04_static193_160x48_f000080__py_spike_cuda__s0043__r01'
    output=ROOT/'experiments/validation'/rid
    if output.exists() or rid in known:raise FileExistsError(output)
    config['output']=str(output)
    artifact=dict(schema='mstd-artifact-v1',id=rid,kind='validation',status='running',stimulus_ids=[SID],
        input_paths=[rel],input_sha256=sha(dat),parent_run_ids=meta['parent_run_ids'],
        notes='Matched static image, same104ms/seed43/model; spike-event/neural-state dumping disabled only.',source_hashes={'tools/run_kitti_static_control.py':sha(Path(__file__))})
    print('OUTPUT:',output.relative_to(ROOT).as_posix(),flush=True)
    try:
        from mstd.simulation import simulate
        report=simulate(config);assert report['complete'] and report['frames']==80
        rows=[json.loads(s) for s in (output/'estimates.jsonl').read_text().splitlines()];stable=rows[20:]
        summary=dict(frames=80,warmup_frames=20,evaluation_frames=len(stable),static_heading_deg=None,
            false_heading_frames=sum(s['bearing_valid'] for s in stable),false_heading_fraction=sum(s['bearing_valid'] for s in stable)/len(stable),
            mt_evidence_mean=float(np.mean([s['mt_evidence'] for s in stable])),mt_evidence_max=max(s['mt_evidence'] for s in stable),
            interpretation='Static translation direction undefined; valid heading is a false positive. No heading MAE calculated.')
        (output/'static_metrics.json').write_text(json.dumps(summary,indent=2));artifact['status']='completed'
        print(json.dumps(summary))
    except BaseException:
        artifact['status']='failed';raise
    finally:
        output.mkdir(parents=True,exist_ok=True);(output/'artifact.json').write_text(json.dumps(artifact,indent=2))
        register(dict(id=rid,kind='validation',path=output.relative_to(ROOT).as_posix(),status=artifact['status'],parent_run_ids=';'.join(meta['parent_run_ids']),stimulus_ids=SID,notes=artifact['notes']))
if __name__=='__main__':main()
