"""Signed 28x8 readout of raw filter spike counts across three scales."""
from dataclasses import dataclass
import hashlib
import json
from math import isfinite
from pathlib import Path
from .data import V1Frame, V1Spec
from .carlsim import iter_bins, read_header
from .io import write_recording


@dataclass(frozen=True)
class Projection:
    name: str
    weights: tuple[tuple[float, ...], ...]
    scale_weights: tuple[float, ...]
    directions_rad: tuple[float, ...]
    gain: float

    def __post_init__(self):
        if len(self.weights) != 28 or any(len(row) != 8 for row in self.weights):
            raise ValueError('projection must have shape 28x8')
        if len(self.scale_weights) != 3 or len(self.directions_rad) != 8:
            raise ValueError('three scales and eight direction angles required')
        values = [v for row in self.weights for v in row]
        values += list(self.scale_weights)+list(self.directions_rad)+[self.gain]
        if not all(isfinite(v) for v in values):
            raise ValueError('finite projection parameters required')
        if self.gain <= 0 or min(self.scale_weights) < 0 or not any(self.scale_weights):
            raise ValueError('positive gain and nonnegative nonzero scale weights required')
        V1Spec(2, 2, self.directions_rad, 1, 1, 0, 0)


def load_projection(path, name='motionProj1'):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    if config.get('schema') != 'v1-filter-projection-v1':
        raise ValueError('unsupported projection schema')
    entry = config['projections'][name]
    return Projection(name, tuple(tuple(row) for row in entry['weights']),
                      tuple(config['scale_weights']), tuple(config['directions_rad']), config['gain'])


def project_counts(spec, counts, width_ms, projection):
    if not isfinite(width_ms) or width_ms <= 0:
        raise ValueError('positive finite bin width required')
    n = spec.width*spec.height
    if len(counts) != 84*n:
        raise ValueError('expected W*H*28*3 filter counts')
    if tuple(spec.directions_rad) != projection.directions_rad:
        raise ValueError('output angles do not match projection')
    output = [0.0]*(8*n)
    for neuron_id, count in enumerate(counts):
        if not isfinite(count) or count < 0:
            raise ValueError('counts must be finite and nonnegative')
        if not count:
            continue
        plane, pixel = divmod(neuron_id, n)
        scale, channel = divmod(plane, 28)
        rate = count*(1000.0/width_ms)*projection.scale_weights[scale]
        for direction, weight in enumerate(projection.weights[channel]):
            output[direction*n+pixel] += rate*weight
    return tuple(max(0.0, value*projection.gain) for value in output)


def convert_spikes(path, output, projection_path, projection_name, duration_ms, bin_ms, intrinsics):
    if type(duration_ms) is not int or duration_ms <= 0 or type(bin_ms) is not int or bin_ms <= 0:
        raise ValueError('positive integer duration and bin width required')
    header = read_header(path)
    if header.depth != 84:
        raise ValueError('raw-filter adapter requires depth=28*3=84')
    projection = load_projection(projection_path, projection_name)
    spec = V1Spec(header.width, header.height, projection.directions_rad, **intrinsics)
    def frames():
        for start, width, counts in iter_bins(path, duration_ms, bin_ms):
            yield V1Frame(start/1000.0, project_counts(spec, counts, width, projection))
    provenance = {
        'source': 'carlsim_spike_filter_projection', 'source_file': Path(path).name,
        'grid': [header.width, header.height, header.depth], 'projection_name': projection_name,
        'projection_config_sha256': hashlib.sha256(Path(projection_path).read_bytes()).hexdigest(),
        'projection': {'weights': projection.weights, 'scale_weights': projection.scale_weights,
                       'gain': projection.gain, 'directions_rad': projection.directions_rad},
        'duration_ms': duration_ms, 'bin_ms': bin_ms,
        'last_frame_duration_s': ((duration_ms-1)%bin_ms+1)/1000.0,
        'rate_available_at': 'bin_end',
        'limitations': ['binned Poisson realization', 'linear projection, not CARLsim E/I dynamics',
                        'direction convention requires empirical calibration'],
    }
    write_recording(output, spec, frames(), provenance)
    return {'output': str(output), 'events': header.event_count,
            'bins': (duration_ms+bin_ms-1)//bin_ms, 'projection': projection_name}
