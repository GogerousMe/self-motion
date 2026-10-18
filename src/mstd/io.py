"""V1-rate-v1: JSON metadata + little-endian float32, [t,d,y,x]."""

import json
import struct
from math import isfinite
from pathlib import Path

from .data import V1Frame, V1Spec


def write_recording(path, spec, frames, provenance):
    """Stream frames into a new directory. Never overwrite a recording."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=False)
    binary, meta_path = directory/'rates.f32', directory/'metadata.json'
    times, previous = [], -1.0
    try:
        with binary.open('xb') as stream:
            for frame in frames:
                frame.validate(spec)
                if frame.timestamp_s <= previous:
                    raise ValueError('timestamps must be strictly increasing')
                block = struct.pack(f'<{spec.frame_size}f', *frame.rates_hz)
                if any(not isfinite(x) for x in struct.unpack(f'<{spec.frame_size}f', block)):
                    raise ValueError('rates exceed finite float32 range')
                stream.write(block)
                times.append(frame.timestamp_s)
                previous = frame.timestamp_s
        if not times:
            raise ValueError('recording must contain frames')
        metadata = {
            'schema': 'v1-rate-v1', 'encoding': 'direction_rates', 'dtype': 'float32-le',
            'layout': 'TDYX', 'units': 'Hz', 'coordinates': 'x-right_y-down',
            'timestamp_reference': 'frame_start', 'sample_hold': 'until_next_timestamp',
            'width': spec.width, 'height': spec.height,
            'directions_rad': list(spec.directions_rad),
            'intrinsics_px': {k: getattr(spec, k) for k in ('fx_px', 'fy_px', 'cx_px', 'cy_px')},
            'timestamps_s': times, 'provenance': provenance,
        }
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    except BaseException:
        # Only exact files we created, no recursive removal.
        binary.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
        directory.rmdir()
        raise


def recording_info(path):
    directory = Path(path)
    meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    required = {"schema": "v1-rate-v1", "encoding": "direction_rates", "dtype": "float32-le",
                "layout": "TDYX", "units": "Hz", "coordinates": "x-right_y-down",
                "timestamp_reference": "frame_start", "sample_hold": "until_next_timestamp"}
    if any(meta.get(k) != v for k, v in required.items()):
        raise ValueError("unsupported V1 recording schema, encoding, layout, units or timing")
    spec = V1Spec(meta["width"], meta["height"], tuple(meta["directions_rad"]), **meta["intrinsics_px"])
    times = meta["timestamps_s"]
    if not times:
        raise ValueError("empty recording")
    size = spec.frame_size * 4
    if (directory/'rates.f32').stat().st_size != size * len(times):
        raise ValueError("binary length does not match metadata")
    previous = -1.0
    for timestamp in times:
        if not isfinite(timestamp) or timestamp < 0 or timestamp <= previous:
            raise ValueError("timestamps must be finite, nonnegative and strictly increasing")
        previous = timestamp
    return spec, meta


def iter_recording_frames(path, spec, meta):
    with (Path(path)/'rates.f32').open('rb') as stream:
        for timestamp in meta['timestamps_s']:
            block = stream.read(spec.frame_size*4)
            if len(block) != spec.frame_size*4:
                raise ValueError('truncated recording')
            frame = V1Frame(timestamp, struct.unpack(f'<{spec.frame_size}f', block))
            frame.validate(spec)
            yield frame
        if stream.read(1):
            raise ValueError('extra recording bytes')


def read_recording(path):
    """Convenience eager API for small recordings."""
    spec, meta = recording_info(path)
    return spec, list(iter_recording_frames(path, spec, meta)), meta
