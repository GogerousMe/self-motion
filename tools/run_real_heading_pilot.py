"""Run a self-contained real-image heading pilot and register its responses."""
import argparse,csv
from datetime import datetime,timedelta,timezone
import hashlib,json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mstd.calibration import inspect_stimulus
from mstd.data import V1Spec
from mstd.model import load_model_config,model_config
from mstd.readout import ReadoutConfig,heading_readout
from mstd.simulation import simulate

SID='kitti04_f000153_f000232__160x48__f000080__dt104ms__gray_resize__v01'
DEFAULT='input/stimuli/natural/kitti/'+SID+'/'+SID+'.dat'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def evaluate(output,folder,meta):
    report=json.loads((output/'model_report.json').read_text(encoding='utf-8'))
    config=json.loads((output/'model_config.json').read_text(encoding='utf-8'))
    states=[json.loads(s) for s in (output/'estimates.jsonl').read_text(encoding='utf-8').splitlines()]
    truth=[json.loads(s) for s in (folder/meta['truth_file']).read_text(encoding='utf-8').splitlines()]
    assert report['complete'] and len(states)==len(truth)==meta['frames']
    spec=V1Spec(meta['width'],meta['height'],tuple(np.arange(8)*np.pi/4),**meta['intrinsics_px'])
    centers=report['model']['mstd_template_centers_px'];cfg=ReadoutConfig(**config['mstd_readout'])
    rows=[]
    for i,(s,t) in enumerate(zip(states,truth)):
        assert s['frame']==t['frame']==i and abs(s['input_timestamp_s']-t['input_time_s'])<1e-8
        decoded=heading_readout(s['activity'],centers,spec,s['winner_index'],True,cfg)
        rows.append(dict(frame=i,expected=t['expected_heading_deg'],winner=decoded['winner_alpha_x_deg'],
                         selected=decoded['alpha_x_deg'],valid=s['bearing_valid']))
        if s['bearing_valid']:assert abs(s['alpha_x_deg']-decoded['alpha_x_deg'])<1e-8
    stable=rows[20:];valid=[r for r in stable if r['valid']]
    metrics=dict(frames=len(rows),warmup_frames=20,evaluation_frames=len(stable),valid_fraction=len(valid)/len(stable),
        readout=dict(config['mstd_readout']),status='exploratory, no predeclared real-video acceptance threshold')
    for mode in ('winner','selected'):
        e=np.array([abs((r[mode]-r['expected']+180)%360-180) for r in stable])
        metrics[mode]=dict(unconditional_mae_deg=float(e.mean()),p95_error_deg=float(np.percentile(e,95)),max_error_deg=float(e.max()),
            valid_mae_deg=float(np.mean([abs((r[mode]-r['expected']+180)%360-180) for r in valid])) if valid else None)
    (output/'heading_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    lines=['# KITTI真实图像航向试运行','',f"稳态有效率：{metrics['valid_fraction']:.1%}；剔除前20帧。",'',
        '| 读出 | 未门控MAE | 95分位误差 | 最大误差 |','|---|---:|---:|---:|']
    for mode in ('winner','selected'):
        r=metrics[mode];lines.append(f"| {mode} | {r['unconditional_mae_deg']:.3f}° | {r['p95_error_deg']:.3f}° | {r['max_error_deg']:.3f}° |")
    lines.extend(['','这是近直行真实场景的探索性试运行，真值为约1s窗口的相机坐标平移方向参考。',
        '筛选按GT，不按模型响应；时间已显式近似均匀化，空间标定随缩放更新。',
        '未套用合成数据2.5°目标，也未进行官方KITTI位姿评测。一个近0°片段不能证明任意航向泛化。',
        '后续需要匹配静态画面控制、更多运动方向和独立片段；GT和深度均不输入模型。'])
    (output/'HEADING_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stimulus',default=DEFAULT)
    p.add_argument('--config',default='configs/model_python_spiking_geometry_quadratic_v01.json')
    p.add_argument('--backend',choices=['numpy','torch'],default='torch');p.add_argument('--device',default='cuda')
    p.add_argument('--seed',type=int,default=43);p.add_argument('--plan-only',action='store_true')
    a=p.parse_args();path=(ROOT/a.stimulus).resolve()
    if not path.is_relative_to(ROOT/'input/stimuli/natural'):p.error('use indexed natural stimulus bundle')
    folder=path.parent;meta=json.loads(path.with_suffix('.dat.json').read_text(encoding='utf-8'))
    info=inspect_stimulus(path);digest=sha(path)
    assert digest==meta['stimulus_sha256'] and all(info[k]==meta[k] for k in ('width','height','frames','channels'))
    assert meta['intrinsics_px']==json.loads((folder/meta['intrinsics_file']).read_text(encoding='utf-8'))
    assert (folder/meta['truth_file']).is_file()
    config=load_model_config(ROOT/a.config)
    config.update(input=dict(kind='stimulus_dat',path=path.relative_to(ROOT).as_posix()),
        backend=a.backend,device=a.device,frame_ms=meta['frame_ms'],intrinsics=meta['intrinsics_px'])
    config['spiking']['seed']=a.seed;config=model_config(config)
    if config['mt_backend']!='python_spiking':p.error('pilot wrapper requires Python spiking MT')
    if a.plan_only:
        print(json.dumps(dict(stimulus=path.relative_to(ROOT).as_posix(),shape=info,frame_ms=config['frame_ms'],
            intrinsics=config['intrinsics'],readout=config['mstd_readout'],backend=a.backend,device=a.device,seed=a.seed),indent=2));return
    from mstd.v1 import ArrayOps
    ArrayOps(a.backend,a.device)
    with (ROOT/'input/registry.csv').open(encoding='utf-8-sig',newline='') as f:reader=csv.DictReader(f);input_fields=reader.fieldnames;indexed=[r for r in reader if r['path']==path.relative_to(ROOT).as_posix()]
    if indexed:assert len(indexed)==1 and indexed[0]['sha256']==digest
    else:
        with (ROOT/'input/registry.csv').open('a',encoding='utf-8',newline='') as f:csv.DictWriter(f,fieldnames=input_fields).writerow(dict(path=path.relative_to(ROOT).as_posix(),width=info['width'],height=info['height'],frames=info['frames'],channels=info['channels'],sha256=digest,scope='reusable'))
    registry=ROOT/'experiments/registry.csv'
    with registry.open(encoding='utf-8-sig',newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;known={r['id'] for r in reader}
    if meta['stimulus_id'] not in known:
        with registry.open('a',encoding='utf-8',newline='') as f:csv.DictWriter(f,fieldnames=fields).writerow(dict(id=meta['stimulus_id'],kind='stimulus',path=folder.relative_to(ROOT).as_posix(),status='completed',stimulus_ids=meta['stimulus_id'],notes='Verified copied real-heading pilot bundle.'))
    stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    tag='py_spike_cuda' if a.device.startswith('cuda') else 'py_spike_cpu'
    run_id=f'{stamp}__kitti04_f000153_f000232_160x48_f000080__{tag}__s{a.seed:04d}__r01'
    output=ROOT/'experiments/validation'/run_id
    config['output']=str(output)
    if output.exists() or run_id in known:raise FileExistsError(output)
    artifact=dict(schema='mstd-artifact-v1',id=run_id,kind='validation',status='running',
        stimulus_ids=[meta['stimulus_id']],input_paths=[path.relative_to(ROOT).as_posix()],input_sha256=digest,parent_run_ids=[],
        notes='Exploratory real-image near-forward heading; execution completion is not accuracy acceptance.',
        command=[sys.executable,*sys.argv])
    print('RUN_ID:',run_id,flush=True);print('OUTPUT:',output.relative_to(ROOT).as_posix(),flush=True)
    try:
        simulate(config)
        evaluate(output,folder,meta)
        artifact['status']='completed'
    except BaseException:
        artifact['status']='failed';raise
    finally:
        output.mkdir(parents=True,exist_ok=True)
        artifact['source_hashes']={s:sha(ROOT/s) for s in ('tools/run_real_heading_pilot.py','src/mstd/readout.py','src/mstd/model.py','src/mstd/tensor_pipeline.py')}
        (output/'artifact.json').write_text(json.dumps(artifact,indent=2),encoding='utf-8')
        with registry.open('a',encoding='utf-8',newline='') as f:csv.DictWriter(f,fieldnames=fields).writerow(dict(id=run_id,kind='validation',path=output.relative_to(ROOT).as_posix(),status=artifact['status'],stimulus_ids=meta['stimulus_id'],notes=artifact['notes']))
    print('HEADING_REPORT:',(output/'HEADING_REPORT.md').relative_to(ROOT).as_posix(),flush=True)


if __name__=='__main__':main()
