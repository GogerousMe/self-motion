"""Streaming CARLsim SpikeMonitor v0.2 reader; not a network-snapshot reader."""
from array import array
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
import struct

SIGNATURE = 206661989


@dataclass(frozen=True)
class SpikeHeader:
    version: float
    width: int
    height: int
    depth: int
    endian: str
    event_count: int

    @property
    def neurons(self):
        return self.width*self.height*self.depth


def read_header(path):
    path = Path(path)
    with path.open('rb') as stream:
        raw = stream.read(20)
    if len(raw) != 20:
        raise ValueError('truncated CARLsim spike header')
    endian = next((e for e in ('<', '>') if struct.unpack(e+'i', raw[:4])[0] == SIGNATURE), None)
    if endian is None:
        raise ValueError('not a CARLsim SpikeMonitor file (wrong signature)')
    _, version, width, height, depth = struct.unpack(endian+'ifiii', raw)
    if not isfinite(version) or abs(version-0.2) > 1e-6:
        raise ValueError('only SpikeMonitor v0.2 is supported')
    if min(width, height, depth) <= 0:
        raise ValueError('invalid neuron grid')
    payload = path.stat().st_size-20
    if payload % 8:
        raise ValueError('truncated int32 time,id event')
    return SpikeHeader(version, width, height, depth, endian, payload//8)


def iter_events(path, header=None, chunk_events=65536):
    if type(chunk_events) is not int or chunk_events <= 0:
        raise ValueError('positive integer chunk size required')
    actual = read_header(path)
    if header is not None and actual != header:
        raise ValueError('spike file header changed')
    header = actual
    previous, seen = -1, 0
    with Path(path).open('rb') as stream:
        stream.seek(20)
        while True:
            block = stream.read(chunk_events*8)
            if not block:
                break
            if len(block) % 8:
                raise ValueError('truncated spike event')
            for time_ms, neuron_id in struct.iter_unpack(header.endian+'ii', block):
                if time_ms < 0 or time_ms < previous:
                    raise ValueError('spike times must be nonnegative and nondecreasing')
                if not 0 <= neuron_id < header.neurons:
                    raise ValueError('spike neuron id outside grid')
                previous = time_ms
                seen += 1
                yield time_ms, neuron_id
    if seen != header.event_count:
        raise ValueError('spike file size changed while reading')


def inspect_spikes(path, max_events=10000):
    if type(max_events) is not int or max_events <= 0:
        raise ValueError('positive integer max_events required')
    header = read_header(path)
    checked, first, last = 0, None, None
    events = iter_events(path, header)
    try:
        for event in events:
            first = event if first is None else first
            last = event
            checked += 1
            if checked >= max_events:
                break
    finally:
        events.close()
    return {'header': asdict(header), 'checked_events': checked,
            'full_event_validation': checked == header.event_count,
            'first_checked_event': first, 'last_checked_event': last,
            'note': 'last checked spike is not the simulation duration'}


def iter_bins(path, duration_ms, bin_ms=50):
    """Half-open windows including silent tail; exhaust to validate whole file."""
    if type(duration_ms) is not int or duration_ms <= 0:
        raise ValueError('positive integer duration_ms required')
    if type(bin_ms) is not int or bin_ms <= 0:
        raise ValueError('positive integer bin_ms required')
    header = read_header(path)
    counts = array('I', [0])*header.neurons
    start = 0
    for time_ms, neuron_id in iter_events(path, header):
        if time_ms >= duration_ms:
            raise ValueError('spike at/after duration; provide complete simulation duration')
        while time_ms >= min(start+bin_ms, duration_ms):
            yield start, min(bin_ms, duration_ms-start), counts
            start += bin_ms
            counts = array('I', [0])*header.neurons
        counts[neuron_id] += 1
    while start < duration_ms:
        yield start, min(bin_ms, duration_ms-start), counts
        start += bin_ms
        if start < duration_ms:
            counts = array('I', [0])*header.neurons
