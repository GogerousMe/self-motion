"""Run an indexed independent self-motion baseline and register its evidence."""
import argparse
import csv
from datetime import datetime,timedelta,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mstd.model import load_model_config
from mstd.calibration import inspect_stimulus

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',required=True,choices=['cloud_static','cloud_forward',
        'cloud_heading_left7','cloud_heading_right7','ground_forward',
        'ground_heading_left10','ground_heading_right10'])
    p.add_argument('--backend',choices=['numpy','torch'],default='torch')
    p.add_argument('--device',default='cuda')
    p.add_argument('--config',default='configs/model_python_spiking.json')
    p.add_argument('--suite',default='input/stimuli/selfmotion/selfmotion_suite__80x80__f000080__dt50ms__14cases__v01')
    p.add_argument('--seed',type=int,default=42,help='neural Poisson seed, independent of scene seed')
    p.add_argument('--frozen-config',help='required for geometry mode; frozen_validity.json from development')
    a=p.parse_args()
    root=ROOT
    if a.backend=='numpy' and a.device!='cpu':p.error('numpy requires --device cpu')
    folder=(root/a.suite).resolve()
    if not folder.is_relative_to(root/'input/stimuli'):p.error('suite must be under input/stimuli')
    folder=folder.relative_to(root)
    config=load_model_config(root/a.config)
    if config['mt_backend']!='python_spiking':p.error('this runner requires Python spiking MT')
    if config['frame_ms']!=50 or config['mstd_template_count']!=9:
        p.error('this validation requires 50ms frames and the 9-candidate heading grid')
    if not 0<=a.seed<2**32:p.error('seed must be in [0, 2**32)')
    frozen=None
    if config['mstd_validity']['mode']=='geometry_support':
        if not a.frozen_config:p.error('geometry mode requires --frozen-config to prevent threshold drift')
        frozen=json.loads((root/a.frozen_config).read_text(encoding='utf-8'))
        if not frozen['development_passed'] or frozen['validity']!=config['mstd_validity']:
            p.error('model validity settings differ from qualified frozen parameters')
    stimulus=folder/(a.case+'.dat');intrinsics=folder/'intrinsics.json'
    with (root/'input/registry.csv').open(encoding='utf-8-sig',newline='') as f:
        indexed=next((r for r in csv.DictReader(f) if r['path']==stimulus.as_posix()),None)
    digest=hashlib.sha256((root/stimulus).read_bytes()).hexdigest()
    if indexed is None or indexed['sha256']!=digest:raise ValueError('stimulus registry/hash mismatch')
    if not (root/intrinsics).is_file():raise FileNotFoundError(intrinsics)
    shape=inspect_stimulus(root/stimulus)
    if (shape['width'],shape['height'],shape['frames'])!=(80,80,80):p.error('expected 80x80, 80 frames')
    stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    backend='py_spike_cuda' if a.device.startswith('cuda') else 'py_spike_cpu'
    suffix=(f"__{config['mstd_validity']['mode']}_s{a.seed:04d}"
            if frozen is not None or a.seed!=42 else '')
    run_id=f'{stamp}__{a.case}_80x80_f000080__{backend}{suffix}__r01'
    output=Path('experiments/validation')/run_id
    path=root/output
    if path.exists():raise FileExistsError(path)
    stimulus_id=folder.name+'__'+a.case
    artifact=dict(schema='mstd-artifact-v1',id=run_id,kind='validation',status='running',
        stimulus_ids=[stimulus_id],input_paths=[stimulus.as_posix()],input_sha256=digest,
        parent_run_ids=[],notes='Independent 80-frame baseline with calibrated camera intrinsics; no task passed claim.')
    command=[sys.executable,str(root/'run_model.py'),'--config',a.config,
        '--backend',a.backend,'--device',a.device,'--stimulus',stimulus.as_posix(),
        '--intrinsics',intrinsics.as_posix(),'--output',output.as_posix(),'--spike-seed',str(a.seed)]
    # Model runner requires a nonexistent output. Store provenance only after its execution.
    print('RUN_ID:',run_id,flush=True)
    print('OUTPUT:',output.as_posix(),flush=True)
    code=1
    try:
        code=subprocess.run(command,cwd=root).returncode
    finally:
        path.mkdir(parents=True,exist_ok=True)
        artifact['status']='completed' if code==0 else 'failed'
        artifact['command']=command
        if frozen is not None:
            artifact['validity_freeze_sha256']=hashlib.sha256((root/a.frozen_config).read_bytes()).hexdigest()
            (path/'frozen_validity.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8')
        (path/'artifact.json').write_text(json.dumps(artifact,indent=2),encoding='utf-8')
        registry=root/'experiments/registry.csv'
        with registry.open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
        if any(r['id']==run_id for r in rows):raise ValueError('duplicate registry run ID')
        with registry.open('a',encoding='utf-8',newline='') as f:
            csv.DictWriter(f,fieldnames=fields).writerow(dict(id=run_id,kind='validation',
                path=output.as_posix(),status=artifact['status'],stimulus_ids=stimulus_id,notes=artifact['notes']))
    if code==0 and frozen is not None:
        analysis=Path('experiments/validation')/(run_id+'__gate_check')
        subprocess.run([sys.executable,str(root/'examples/check_mstd_validity.py'),
            '--frozen-config',a.frozen_config,'--parent-run',run_id,'--output',analysis.as_posix()],
            cwd=root,check=True)
        print('GATE_REPORT:',(analysis/'REPORT.md').as_posix(),flush=True)
    raise SystemExit(code)

if __name__=='__main__':main()
