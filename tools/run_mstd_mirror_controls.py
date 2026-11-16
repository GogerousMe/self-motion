"""Run six frozen mirror controls, register them, and summarize MSTd and layers."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mstd.calibration import inspect_stimulus
from mstd.mirror_controls import pair_symmetry
from mstd.model import load_model_config

SUITE='input/stimuli/selfmotion/mstd_mirror_s0211__80x80__f000080__dt50ms__2cases__v01'
FREEZE='experiments/validation/20261109__mstd_geometry_gate_80x80_f000080__py_rate_cpu__r01/frozen_validity.json'
CASES=('cloud_heading_left7','cloud_heading_right7')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def commands(suite,freeze,backend,device,seeds):
    return [(seed,case,[sys.executable,str(ROOT/'tools/run_selfmotion_baseline.py'),
        '--case',case,'--suite',suite,'--config','configs/model_python_spiking_geometry_v01.json',
        '--frozen-config',freeze,'--seed',str(seed),'--backend',backend,'--device',device])
        for seed in seeds for case in CASES]


def summarize_runs(batch,runs):
    rows=[]; sequences={}
    for item in runs:
        run=ROOT/item['output']
        config=json.loads((run/'model_config.json').read_text(encoding='utf-8'))
        report=json.loads((run/'model_report.json').read_text(encoding='utf-8'))
        assert report['complete'] and report['frames']==80 and config['spiking']['seed']==item['seed']
        state=[json.loads(s) for s in (run/'estimates.jsonl').read_text(encoding='utf-8').splitlines()]
        assert len(state)==80
        stable=state[20:]
        predicted=np.array([(s['winner_index']-4)*5 for s in stable],dtype=float)
        target=-7 if item['case']==CASES[0] else 7
        error=predicted-target
        valid=np.array([s['bearing_valid'] for s in stable])
        check=ROOT/(item['output']+'__gate_check')/'metrics.json'
        diagnostic=json.loads(check.read_text(encoding='utf-8'))['summary'][0]
        assert abs(diagnostic['unconditional_winner_mae_deg']-float(np.abs(error).mean()))<1e-9
        rows.append(dict(case=item['case'],seed=item['seed'],run_id=item['run_id'],expected_heading_deg=target,
            valid_fraction=float(valid.mean()),mean_heading_deg=float(predicted.mean()),
            mean_signed_error_deg=float(error.mean()),mae_deg=float(np.abs(error).mean()),
            p95_error_deg=float(np.percentile(np.abs(error),95)),max_error_deg=float(np.abs(error).max()),
            diagnostic_passed=diagnostic['diagnostic_passed']))
        sequences[(item['seed'],item['case'])]=predicted
    paired=[]
    for seed in sorted({x['seed'] for x in runs}):
        if all((seed,c) in sequences for c in CASES):
            paired.append(dict(seed=seed,**pair_symmetry(sequences[(seed,CASES[0])],sequences[(seed,CASES[1])])))
    result=dict(cases=rows,pairs=paired,warmup_frames=20,evaluation_frames=60,
                all_six_targets_passed=(len(rows)==6 and all(r['diagnostic_passed'] for r in rows)),
                interpretation='Mirror bias is a descriptive direction asymmetry, not proof of a neural mechanism. One scene, three neural seeds.')
    dump(batch/'summary.json',result)
    with (batch/'summary.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# ±7°镜像多种子对照','',
        '每个方向80帧，排除前20帧。MSTd仍输出5°网格航向；以下误差使用未门控竞争赢家，未换用插值结果。','',
        '| 神经种子 | 真值方向 | 有效率 | 平均预测 | 平均有符号误差 | MAE | 95分位误差 | 本轮目标 |',
        '|---|---|---:|---:|---:|---:|---:|---|']
    for r in rows:
        lines.append(f"| {r['seed']} | {r['expected_heading_deg']}° | {r['valid_fraction']:.1%} | {r['mean_heading_deg']:.3f}° | {r['mean_signed_error_deg']:.3f}° | {r['mae_deg']:.3f}° | {r['p95_error_deg']:.1f}° | {r['diagnostic_passed']} |")
    lines.extend(['','| 神经种子 | 共同方向偏移均值 | 镜像和的平均绝对值 |','|---|---:|---:|'])
    for r in paired:
        lines.append(f"| {r['seed']} | {r['mean_common_bias_deg']:.3f}° | {r['mean_absolute_mirror_deviation_deg']:.3f}° |")
    lines.extend(['','共同方向偏移=平均(左预测+右预测)/2。理想左右对称时为0；量化、脉冲波动和动态状态均可能造成非零。',
        '同种子未强制镜像脉冲样本，不要求逐帧预测严格相反。三种子仅提供诊断线索，不能单独证明固定偏置或普适精度。',
        '目标沿用：有效率≥95%、未门控MAE≤2.5°。执行完成与目标通过分别记录；保留所有失败结果。',
        '分层报告见本批次artifact中的layer_analysis_path；所有原生输出位置见batch.json。'])
    (batch/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=['numpy','torch'],default='torch')
    p.add_argument('--device',default='cuda')
    p.add_argument('--suite',default=SUITE)
    p.add_argument('--frozen-config',default=FREEZE)
    p.add_argument('--seeds',nargs=3,type=int,default=[43,44,45])
    p.add_argument('--plan-only',action='store_true',help='validate files and print commands; do not run or write outputs')
    a=p.parse_args()
    if len(set(a.seeds))!=3 or any(not 0<=s<2**32 for s in a.seeds):p.error('provide three distinct seeds in [0,2**32)')
    if a.backend=='numpy' and a.device!='cpu':p.error('numpy requires --device cpu')
    folder=(ROOT/a.suite).resolve()
    if not folder.is_relative_to(ROOT/'input/stimuli/selfmotion'):p.error('suite must be under input/stimuli/selfmotion')
    suite=folder.relative_to(ROOT).as_posix()
    frozen_path=(ROOT/a.frozen_config).resolve()
    frozen=json.loads(frozen_path.read_text(encoding='utf-8'))
    config=load_model_config(ROOT/'configs/model_python_spiking_geometry_v01.json')
    if not frozen['development_passed'] or config['mstd_validity']!=frozen['validity']:
        p.error('configured gate differs from qualified frozen parameters')
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('mirror_pair',{}).get('pixel_relation')!='left[t,y,x] == right[t,y,79-x]':
        p.error('expected the exact mirror-control protocol')
    assert manifest['validity_freeze_sha256']==sha(frozen_path)
    assert manifest['intrinsics_px']==json.loads((folder/'intrinsics.json').read_text(encoding='utf-8'))
    with (ROOT/'input/registry.csv').open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);input_fields=reader.fieldnames;index=list(reader)
    missing=[]
    arrays={}
    for case in CASES:
        path=folder/(case+'.dat'); info=inspect_stimulus(path)
        assert (info['width'],info['height'],info['frames'],info['channels'])==(80,80,80,1)
        item=next(r for r in manifest['runs'] if r['name']==case)
        assert sha(path)==item['stimulus_sha256']
        relative=path.relative_to(ROOT).as_posix()
        indexed=[r for r in index if r['path']==relative]
        if indexed:
            if len(indexed)!=1 or indexed[0]['sha256']!=sha(path):
                raise ValueError('existing stimulus registry entry disagrees with verified DAT')
        else:
            missing.append(dict(path=relative,width=80,height=80,frames=80,channels=1,sha256=sha(path),scope='reusable'))
        arrays[case]=np.frombuffer(path.read_bytes()[25:],dtype=np.uint8).reshape(80,80,80)
    assert np.array_equal(arrays[CASES[0]],arrays[CASES[1]][:,:,::-1])
    plan=commands(suite,a.frozen_config,a.backend,a.device,a.seeds)
    if a.plan_only:
        print(json.dumps(dict(cases=2,runs=6,total_frames=480,input_registry_rows_to_add=len(missing),
            commands=[dict(seed=s,case=c,argv=v) for s,c,v in plan]),indent=2))
        return
    # Fail before creating a batch if the selected execution device is unavailable.
    from mstd.v1 import ArrayOps
    ArrayOps(a.backend,a.device)
    # Import the copied bundle idempotently after verifying its pixels/hashes.
    # Preserve all pre-existing server registry records.
    if missing:
        with (ROOT/'input/registry.csv').open('a',encoding='utf-8',newline='') as f:
            csv.DictWriter(f,fieldnames=input_fields).writerows(missing)
    with (ROOT/'experiments/registry.csv').open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);registry_fields=reader.fieldnames;bundle_registered=any(r['id']==folder.name for r in reader)
    if not bundle_registered:
        bundle_artifact=json.loads((folder/'artifact.json').read_text(encoding='utf-8'))
        assert bundle_artifact['id']==folder.name
        with (ROOT/'experiments/registry.csv').open('a',encoding='utf-8',newline='') as f:
            csv.DictWriter(f,fieldnames=registry_fields).writerow(dict(id=folder.name,kind='stimulus_bundle',
                path=folder.relative_to(ROOT).as_posix(),status='completed',stimulus_ids=';'.join(bundle_artifact['stimulus_ids']),
                notes='Copied mirror bundle locally verified and registered by batch runner.'))
    stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    tag='py_spike_cuda' if a.device.startswith('cuda') else 'py_spike_cpu'
    batch_id=f'{stamp}__mirror7_s0211_80x80_f000480__{tag}__r01'
    batch=ROOT/'experiments/validation'/batch_id
    batch.mkdir(parents=True,exist_ok=False)
    artifact=dict(schema='mstd-artifact-v1',id=batch_id,kind='validation',status='running',
        stimulus_ids=[folder.name+'__'+c for c in CASES],input_paths=[suite+'/'+c+'.dat' for c in CASES],
        input_sha256={suite+'/'+r['stimulus']:r['stimulus_sha256'] for r in manifest['runs']},
        parent_run_ids=[],validity_freeze_sha256=sha(frozen_path),
        notes='Exact pixel mirror, three neural seeds; developmental diagnostic batch, not final blind acceptance.')
    dump(batch/'artifact.json',artifact)
    runs=[]; success=False
    try:
        for seed,case,command in plan:
            run_id=None
            with (batch/f'{case}_s{seed:04d}.log').open('x',encoding='utf-8') as log:
                child=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                       text=True,encoding='utf-8',errors='replace')
                try:
                    for line in child.stdout:
                        print(line,end='',flush=True);log.write(line)
                        if line.startswith('RUN_ID:'):run_id=line.partition(':')[2].strip()
                    return_code=child.wait()
                except BaseException:
                    child.terminate();child.wait();raise
                finally:
                    child.stdout.close()
            if run_id:
                runs.append(dict(case=case,seed=seed,run_id=run_id,output='experiments/validation/'+run_id,
                                 return_code=return_code))
                artifact['parent_run_ids'].append(run_id)
            dump(batch/'batch.json',dict(runs=runs,planned_runs=6,neural_seeds=a.seeds,suite=suite))
            if return_code!=0 or not run_id:
                raise RuntimeError(f'run failed: {case} seed{seed}; see batch log')
        summarize_runs(batch,runs)
        layer=Path('outputs/comparisons/selfmotion')/(batch_id+'__layers__v01')
        command=[sys.executable,str(ROOT/'examples/inspect_mstd_cloud_holdout.py'),'--suite',suite,'--output',layer.as_posix()]
        for r in runs:command+=['--parent-run',r['run_id']]
        with (batch/'layer_analysis.log').open('x',encoding='utf-8') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        artifact['layer_analysis_path']=layer.as_posix()
        success=True
    finally:
        artifact['status']='completed' if success else 'partial'
        artifact['source_hashes']={s:sha(ROOT/s) for s in ('tools/run_mstd_mirror_controls.py','tools/run_selfmotion_baseline.py',
            'examples/inspect_mstd_cloud_holdout.py','configs/model_python_spiking_geometry_v01.json',
            'src/mstd/mirror_controls.py','src/mstd/model.py','src/mstd/tensor_pipeline.py','src/mstd/spiking_mt.py','src/mstd/neural.py')}
        dump(batch/'artifact.json',artifact)
        registry=ROOT/'experiments/registry.csv'
        with registry.open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f);fields=reader.fieldnames
            assert not any(r['id']==batch_id for r in reader)
        with registry.open('a',encoding='utf-8',newline='') as f:
            csv.DictWriter(f,fieldnames=fields).writerow(dict(id=batch_id,kind='validation',path=batch.relative_to(ROOT).as_posix(),
                status=artifact['status'],parent_run_ids=';'.join(artifact['parent_run_ids']),stimulus_ids=';'.join(artifact['stimulus_ids']),notes=artifact['notes']))
    print('BATCH_OUTPUT:',batch.relative_to(ROOT).as_posix(),flush=True)
    print('BATCH_REPORT:',(batch/'REPORT.md').relative_to(ROOT).as_posix(),flush=True)
    print('LAYER_ANALYSIS:',artifact['layer_analysis_path'],flush=True)


if __name__=='__main__':main()
