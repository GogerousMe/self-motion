"""Run all eight independently reset calibration DATs through Python V1/MT."""
import argparse
import csv
import json
from pathlib import Path

from mstd.calibration import file_sha256,inspect_stimulus
from mstd.v1_recording import dump_json
from mstd.v1_workflow import run_frontend


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest')
    parser.add_argument('--output',required=True)
    parser.add_argument('--projection-config',required=True)
    parser.add_argument('--projection',default='motionProj1')
    parser.add_argument('--backend',choices=['numpy','torch'],default='numpy')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--save-raw',action='store_true')
    parser.add_argument('--save-mt',action='store_true')
    args=parser.parse_args()
    manifest_path=Path(args.manifest)
    meta=json.loads(manifest_path.read_text(encoding='utf-8'))
    if meta.get('schema')!='v1-direction-calibration-v1' or meta.get('coordinates')!='x-right_y-down':
        raise ValueError('expected independent eight-direction calibration manifest')
    if meta['frames']*meta['frame_ms']!=meta['duration_ms'] or meta['warmup_ms']%meta['frame_ms']:
        raise ValueError('invalid calibration timing')
    runs=meta['runs']
    if len(runs)!=8 or {r['expected_direction_deg']%360 for r in runs}!=set(range(0,360,45)):
        raise ValueError('eight distinct directions required')
    for run in runs:
        stimulus=manifest_path.parent/run['stimulus']
        info=inspect_stimulus(stimulus)
        if any(info[k]!=meta[k] for k in ('width','height','frames')):
            raise ValueError('manifest/stimulus dimension mismatch')
        if file_sha256(stimulus)!=run['stimulus_sha256']:
            raise ValueError('manifest/stimulus checksum mismatch')
        if Path(run['name']).name!=run['name'] or run['name'] in ('.','..'):
            raise ValueError('run name must be a plain directory name')
    if len({r['name'] for r in runs})!=8:
        raise ValueError('duplicate run names')
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for run in runs:
        report=run_frontend(manifest_path.parent/run['stimulus'],output/run['name'],
                            args.projection_config,args.backend,args.device,meta['frame_ms'],
                            save_raw=args.save_raw,save_mt=args.save_mt,projection_name=args.projection,
                            warmup_frames=meta['warmup_ms']//meta['frame_ms'],
                            expected_direction_deg=run['expected_direction_deg'])
        check=report['direction_check']
        rows.append({'name':run['name'],**check,'wall_fps':report['wall_fps']})
    with (output/'directions.csv').open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    report={'schema':'python-v1-direction-suite-v1','passed':all(r['passed'] for r in rows),
            'manifest_sha256':file_sha256(manifest_path),'runs':rows,
            'note':'direction tuning alone does not prove C++ numerical equivalence or MSTd accuracy'}
    dump_json(output/'suite_report.json',report)
    print(json.dumps({'passed':report['passed'],'output':str(output)},indent=2))
    return 0 if report['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
