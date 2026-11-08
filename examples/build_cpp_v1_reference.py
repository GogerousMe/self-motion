"""Build an isolated CUDA V1 capture executable from the pinned source snapshot."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from mstd.calibration import file_sha256
from mstd.v1 import SOURCE_SHA256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion-energy-root',default='reference_v1',help='folder containing src/ and inc/')
    parser.add_argument('--cuda-root',default='/usr/local/cuda-11.4')
    parser.add_argument('--cuda-samples-include',help='directory containing helper_cuda.h')
    parser.add_argument('--arch',default='sm_86',help='RTX A5000 default')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if not sys.platform.startswith('linux'):
        raise RuntimeError('actual CUDA reference build is intended for Linux server')
    root=Path(args.motion_energy_root).resolve()
    source=root/'src/motion_energy.cu'
    include=root/'inc'
    if file_sha256(source)!=SOURCE_SHA256:
        raise ValueError('source differs from pinned Python profile; inspect differences first')
    cuda=Path(args.cuda_root).resolve()
    samples=Path(args.cuda_samples_include).resolve() if args.cuda_samples_include else cuda/'samples/common/inc'
    if not (samples/'helper_cuda.h').is_file():
        raise FileNotFoundError('helper_cuda.h not found; supply --cuda-samples-include from existing CARLsim build')
    compiler=cuda/'bin/nvcc'
    if not compiler.is_file():raise FileNotFoundError(compiler)
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    # Ensure nvcc can find its runtime compiler tools without changing global environment.
    environment=dict(os.environ)
    environment['PATH']=str(cuda/'bin')+os.pathsep+environment.get('PATH','')
    obj=output/'motion_energy.o'
    library=output/'libME_reference.a'
    executable=output/'export_v1_continuous'
    exporter=Path(__file__).resolve().parent/'export_v1_continuous.cpp'
    commands=[
        [str(compiler),'-std=c++11','-O3','-arch='+args.arch,'-I'+str(include),'-I'+str(samples),
         '-c',str(source),'-o',str(obj)],
        ['ar','rcs',str(library),str(obj)],
        [str(compiler),'-std=c++11','-O2','-I'+str(include),str(exporter),str(library),'-o',str(executable)],
    ]
    for command in commands:
        print('[build]', ' '.join(command),flush=True)
        subprocess.run(command,check=True,env=environment)
    meta={'source_sha256':file_sha256(source),'header_sha256':file_sha256(include/'motion_energy.h'),
          'library_sha256':file_sha256(library),'exporter_sha256':file_sha256(exporter),
          'commands':commands,'source':str(source),'library':str(library),'executable':str(executable),
          'nvcc_version':subprocess.check_output([str(compiler),'--version'],text=True,env=environment)}
    (output/'build.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(executable)


if __name__=='__main__':main()
