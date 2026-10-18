import argparse
import json
from pathlib import Path

from .io import recording_info, iter_recording_frames, write_recording
from .mt import encode_mt
from .mstd import MSTd, horizontal_centers
from .synthetic import default_spec, radial_frame


def main(argv=None):
    parser = argparse.ArgumentParser(description="MT-MSTd direction-rate reference model")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="run synthetic radial-response smoke experiment")
    demo.add_argument("--output", default="outputs/demo")
    run = commands.add_parser("run", help="process a validated V1-rate-v1 recording")
    run.add_argument("recording")
    run.add_argument("--output", required=True)
    run.add_argument("--last-frame-duration-s", type=float, required=True)
    inspect = commands.add_parser('inspect-spikes', help='inspect a CARLsim SpikeMonitor DAT')
    inspect.add_argument('spikes')
    inspect.add_argument('--max-events', type=int, default=10000)
    convert = commands.add_parser('convert-spikes', help='stream 84-channel filter spikes to eight direction rates')
    convert.add_argument('spikes')
    convert.add_argument('--output', required=True)
    convert.add_argument('--projection-config', required=True)
    convert.add_argument('--projection', default='motionProj1')
    convert.add_argument('--duration-ms', type=int, required=True)
    convert.add_argument('--bin-ms', type=int, default=50)
    convert.add_argument('--intrinsics', required=True, help='JSON with fx_px,fy_px,cx_px,cy_px')
    generate = commands.add_parser('generate-calibration', help='write eight known-direction C++ VisualStimulus DATs')
    generate.add_argument('--output', required=True)
    generate.add_argument('--width', type=int, default=80)
    generate.add_argument('--height', type=int, default=80)
    generate.add_argument('--frames', type=int, default=80)
    generate.add_argument('--frame-ms', type=int, default=50)
    generate.add_argument('--warmup-frames', type=int, default=20)
    generate.add_argument('--speed', type=float, default=1.5)
    generate.add_argument('--kind', choices=['grating','texture'], default='grating')
    generate.add_argument('--seed', type=int, default=0)
    analyze = commands.add_parser('analyze-calibration', help='compare actual V1 responses against a stimulus manifest')
    analyze.add_argument('manifest')
    analyze.add_argument('--projection-config', required=True)
    analyze.add_argument('--projection', default='motionProj1')
    analyze.add_argument('--output', required=True)
    analyze.add_argument('--border', type=int, default=4)
    analyze.add_argument('--tolerance-deg', type=float, default=22.5)
    analyze.add_argument('--min-resultant', type=float, default=.1)
    args = parser.parse_args(argv)
    if args.command == 'generate-calibration':
        from .calibration import generate_suite
        print(json.dumps(generate_suite(args.output,args.width,args.height,args.frames,args.frame_ms,
                         args.warmup_frames,args.speed,args.kind,args.seed)))
        return
    if args.command == 'analyze-calibration':
        from .calibration import analyze_calibration
        import csv
        report = analyze_calibration(args.manifest,args.projection_config,args.projection,
                                     args.border,args.tolerance_deg,args.min_resultant)
        output = Path(args.output)
        output.mkdir(parents=True,exist_ok=False)
        (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        fields = ['name','status','expected_direction_deg','winner_direction_deg',
                  'winner_error_deg','resultant','responsive','within_tolerance']
        with (output/'directions.csv').open('w',newline='',encoding='utf-8') as stream:
            writer = csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
            writer.writeheader()
            writer.writerows(report['runs'])
        print(json.dumps({key:report[key] for key in ('measured_runs','pending_runs','calibration_passed')}))
        return
    if args.command == 'inspect-spikes':
        from .carlsim import inspect_spikes
        print(json.dumps(inspect_spikes(args.spikes, args.max_events), indent=2))
        return
    if args.command == 'convert-spikes':
        from .projection import convert_spikes
        intrinsics = json.loads(Path(args.intrinsics).read_text(encoding='utf-8'))
        print(json.dumps(convert_spikes(args.spikes, args.output, args.projection_config,
                         args.projection, args.duration_ms, args.bin_ms, intrinsics)))
        return
    output = Path(args.output)
    if args.command == "demo":
        spec = default_spec()
        duration = 0.05
        frames = [radial_frame(spec, i*duration, 0.0) for i in range(20)]
        output.mkdir(parents=True, exist_ok=False)
        write_recording(output / "recording", spec, frames, {"source": "synthetic_direction_rates", "seed": None})
        recording = output/'recording'
        spec, metadata = recording_info(recording)
    else:
        from math import isfinite
        duration = args.last_frame_duration_s
        if not isfinite(duration) or duration <= 0:
            parser.error("last-frame-duration-s must be positive and finite")
        recording = Path(args.recording)
        spec, metadata = recording_info(recording)
        recorded_duration = metadata.get('provenance', {}).get('last_frame_duration_s')
        if recorded_duration is not None and abs(duration-recorded_duration) > 1e-9:
            parser.error('last-frame duration disagrees with converted recording metadata')
        output.mkdir(parents=True, exist_ok=False)
    angles = list(range(-20, 21, 5))
    model = MSTd(spec, horizontal_centers(spec, angles))
    rows = []
    times = metadata['timestamps_s']
    for index, frame in enumerate(iter_recording_frames(recording, spec, metadata)):
        interval = times[index+1]-frame.timestamp_s if index+1 < len(times) else duration
        result = model.step(encode_mt(spec, frame), interval)
        result.update({"input_timestamp_s": frame.timestamp_s,
                       "estimate_timestamp_s": frame.timestamp_s+interval})
        rows.append(result)
    (output / "estimates.json").write_text(json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8")
    config = {"model": "radial_cd_inspired_reference_v0.1", "pool_radius": 1,
              "horizontal_template_angles_deg": angles, "dynamics": model.config.__dict__,
              "last_frame_duration_s": duration, "input_metadata": metadata,
              "limitations": ["pure-translation radial model", "no calibrated confidence", "not a paper reproduction"]}
    (output / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    last = rows[-1]
    print(json.dumps({"frames": len(rows), "valid": last["valid"],
                      "alpha_x_deg": last["alpha_x_deg"], "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
