"""Convert naturally ordered image files to a CARLsim/Python VisualStimulus DAT."""
import argparse
import json
from math import isfinite
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from mstd.image_sequence import convert_images


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='image folder')
    parser.add_argument('--output', required=True, help='new DAT file; companion .dat.json records frame order/timing')
    parser.add_argument('--mode', choices=['gray', 'rgb'], default='gray')
    timing = parser.add_mutually_exclusive_group()
    timing.add_argument('--frame-ms', type=float, help='model interval; default 50 ms, not inferred from filenames')
    timing.add_argument('--fps', type=float, help='original sequence FPS, as an alternative to --frame-ms')
    parser.add_argument('--geometry', choices=['native', 'resize', 'pad'], default='native')
    parser.add_argument('--size', nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'))
    parser.add_argument('--pattern', default='*', help='optional file glob, e.g. *.jpg')
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--progress-every', type=int, default=20)
    args = parser.parse_args(argv)
    if args.fps is not None and (not isfinite(args.fps) or args.fps <= 0):
        parser.error('positive finite --fps required')
    interval = 1000/args.fps if args.fps is not None else args.frame_ms if args.frame_ms is not None else 50.
    result = convert_images(args.input, args.output, mode=args.mode, frame_ms=interval,
                            geometry=args.geometry, size=args.size, pattern=args.pattern,
                            max_frames=args.max_frames, progress_every=args.progress_every)
    print(json.dumps({k: result[k] for k in ('width', 'height', 'channels', 'frames', 'frame_ms', 'geometry')}, indent=2))
    print(f'[done] {args.output}; metadata: {args.output}.json')
    return result


if __name__ == '__main__':
    main()
