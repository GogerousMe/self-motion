"""python -m mstd.v1_cli: optional NumPy/PyTorch frontend tools."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description='Source-compatible Python MotionEnergy V1 tools')
    commands = parser.add_subparsers(dest='command',required=True)
    run = commands.add_parser('run',help='stream image DAT through Python V1 -> MT (optional MSTd)')
    run.add_argument('stimulus')
    run.add_argument('--output',required=True)
    run.add_argument('--projection-config',required=True)
    run.add_argument('--projection',default='motionProj1')
    run.add_argument('--backend',choices=['numpy','torch'],default='numpy')
    run.add_argument('--device',default='cpu',help='cpu or cuda[:index]; no implicit fallback')
    run.add_argument('--frame-ms',type=float,default=50)
    run.add_argument('--max-frames',type=int)
    run.add_argument('--save-raw',action='store_true')
    run.add_argument('--save-mt',action='store_true')
    run.add_argument('--radius',type=int,default=1)
    run.add_argument('--border',type=int,default=4)
    run.add_argument('--warmup-frames',type=int,default=20)
    run.add_argument('--expected-direction-deg',type=float)
    run.add_argument('--intrinsics',help='real camera intrinsics JSON; enables MSTd and legacy rate recording')
    run.add_argument('--progress-every',type=int,default=20)
    compare = commands.add_parser('compare',help='compare paired 84-channel continuous recordings and MT stages')
    compare.add_argument('reference')
    compare.add_argument('candidate')
    compare.add_argument('--output',required=True)
    compare.add_argument('--projection-config',required=True)
    compare.add_argument('--projection',default='motionProj1')
    compare.add_argument('--atol',type=float,default=5e-4)
    compare.add_argument('--rtol',type=float,default=5e-4)
    compare.add_argument('--radius',type=int,default=1)
    register = commands.add_parser('register-cpp',help='bind C++ capture to exact stimulus/source/library hashes')
    register.add_argument('capture')
    register.add_argument('--stimulus',required=True)
    register.add_argument('--source',required=True)
    register.add_argument('--library',required=True)
    environment = commands.add_parser('environment',help='inspect Python, NumPy and optional torch/CUDA')
    args = parser.parse_args(argv)
    if args.command=='environment':
        import platform
        import numpy as np
        result = {'python':platform.python_version(),'numpy':np.__version__}
        try:
            import torch
            result.update({'torch':torch.__version__,'torch_cuda':torch.version.cuda,
                           'cuda_available':torch.cuda.is_available()})
            if result['cuda_available']:
                result['gpu'] = torch.cuda.get_device_name(0)
        except ImportError:
            result['torch'] = None
        print(json.dumps(result,indent=2))
    elif args.command=='register-cpp':
        from .v1_recording import register_cpp_capture
        result = register_cpp_capture(args.capture,args.stimulus,args.source,args.library)
        print(json.dumps({'registered':args.capture,'frames':result['frames']},indent=2))
    elif args.command=='compare':
        from .v1_workflow import compare_raw
        report = compare_raw(args.reference,args.candidate,args.output,args.projection_config,
                             args.projection,args.atol,args.rtol,args.radius)
        print(json.dumps({k:report[k] for k in ('passed','raw_max_abs','raw_rmse','stage_max_abs')},indent=2))
        return 0 if report['passed'] else 2
    else:
        from .v1_workflow import run_frontend
        intrinsics = json.loads(Path(args.intrinsics).read_text(encoding='utf-8')) if args.intrinsics else None
        report = run_frontend(args.stimulus,args.output,args.projection_config,args.backend,args.device,
                              args.frame_ms,args.max_frames,args.save_raw,args.save_mt,args.projection,
                              args.radius,args.border,args.warmup_frames,args.expected_direction_deg,
                              intrinsics,args.progress_every)
        print(json.dumps({k:report[k] for k in ('frames','wall_seconds','wall_fps','direction_check','mstd_enabled')},indent=2))
        if args.expected_direction_deg is not None and not report['direction_check']['passed']:
            return 2
    return 0


if __name__=='__main__':
    raise SystemExit(main())
