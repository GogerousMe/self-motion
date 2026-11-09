"""Check the actual tensor MSTd with ideal fields; no image/V1/spiking accuracy claim."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from mstd.synthetic import default_spec, radial_frame
from mstd.mt import encode_mt
from mstd.mstd import horizontal_centers
from mstd.tensor_pipeline import TensorMSTd
from mstd.v1 import ArrayOps

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    out=Path(a.output).resolve()
    if not out.is_relative_to(root/'experiments/validation'):
        p.error('output must be inside experiments/validation')
    out.mkdir(parents=True,exist_ok=False)
    spec=default_spec(); op=ArrayOps(); results=[]
    for heading in (0,-10,10,None):
        model=TensorMSTd(op,spec,horizontal_centers(spec,range(-20,21,5)))
        field=encode_mt(spec,radial_frame(spec,0,heading or 0))
        vectors=np.array([field.vx,field.vy],dtype=np.float32).reshape(2,spec.height,spec.width)
        if heading is None:vectors[:]=0
        trace=[]
        for i in range(2000):
            result=model.step({'vectors':vectors,'evidence':field.evidence if heading is not None else 0.},.001)
            if (i+1)%50==0:trace.append(result)
        passed=(not result['valid']) if heading is None else result['valid'] and abs(result['alpha_x_deg']-heading)<1e-5
        results.append(dict(expected_heading_deg=heading,passed=bool(passed),trace=trace))
    report=dict(passed=all(r['passed'] for r in results),cases=results,
        limitation='Ideal radial direction evidence only; not V1/MT or end-to-end validation.')
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    artifact=dict(schema='mstd-artifact-v1',id=out.name,kind='validation',status='completed',
        stimulus_ids=[],input_paths=[],parent_run_ids=[],notes=report['limitation'])
    (out/'artifact.json').write_text(json.dumps(artifact,indent=2),encoding='utf-8')
    registry=root/'experiments/registry.csv'
    with registry.open(encoding='utf-8-sig',newline='') as f:fields=next(csv.reader(f))
    with registry.open('a',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writerow(dict(id=out.name,kind='validation',path=out.relative_to(root).as_posix(),status='completed',notes=report['limitation']))
    print(json.dumps({'passed':report['passed'],'report':str(out/'report.json')}))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()
