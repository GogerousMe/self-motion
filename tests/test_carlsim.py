import json
import math
from pathlib import Path
import struct
import tempfile
import unittest

from mstd.carlsim import inspect_spikes, iter_bins, iter_events, read_header, SIGNATURE
from mstd.data import V1Spec
from mstd.projection import Projection, convert_spikes, load_projection, project_counts
from mstd.io import read_recording


def fixture(path, events=(), endian='<', depth=84):
    path.write_bytes(struct.pack(endian+'ifiii', SIGNATURE, 0.2, 2, 2, depth)
                     + b''.join(struct.pack(endian+'ii', *event) for event in events))


class SpikeTests(unittest.TestCase):
    def test_both_byte_orders_and_chunk_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'spikes.dat'
            events = [(0, 0), (0, 335), (50, 3), (99, 4)]
            for endian in ['<', '>']:
                fixture(path, events, endian)
                self.assertEqual(read_header(path).neurons, 336)
                self.assertEqual(list(iter_events(path, chunk_events=1)), events)
                info = inspect_spikes(path, 2)
                self.assertFalse(info['full_event_validation'])
                self.assertEqual(info['checked_events'], 2)

    def test_half_open_bins_partial_width_and_silent_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'spikes.dat'
            fixture(path, [(0, 0), (49, 0), (50, 1), (99, 2), (100, 3)])
            bins = list(iter_bins(path, 151, 50))
            self.assertEqual([b[:2] for b in bins], [(0, 50), (50, 50), (100, 50), (150, 1)])
            self.assertEqual([sum(b[2]) for b in bins], [2, 2, 1, 0])
            fixture(path)
            self.assertEqual(len(list(iter_bins(path, 100))), 2)

    def test_malformed_events_headers_and_duration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'spikes.dat'
            for events in [[(1, 0), (0, 1)], [(-1, 0)], [(0, -1)], [(0, 336)]]:
                fixture(path, events)
                with self.assertRaises(ValueError):
                    list(iter_events(path))
            fixture(path, [(100, 0)])
            with self.assertRaises(ValueError):
                list(iter_bins(path, 100))
            path.write_bytes(path.read_bytes()+b'x')
            with self.assertRaises(ValueError):
                read_header(path)
            path.write_bytes(b'not a header')
            with self.assertRaises(ValueError):
                read_header(path)

    def test_signed_projection_scale_and_pixel_layout(self):
        weights = [[0.0]*8 for _ in range(28)]
        weights[0][0], weights[1][0], weights[0][1] = 2, -3, -2
        projection = Projection('test', tuple(map(tuple, weights)), (1, 2, 3),
                                tuple(-i*math.pi/4 for i in range(8)), 1)
        spec = V1Spec(2, 2, projection.directions_rad, 2, 2, .5, .5)
        counts = [0]*336
        counts[0], counts[4], counts[28*4+1] = 2, 1, 1
        rates = project_counts(spec, counts, 50, projection)
        self.assertEqual(rates[0], 20)  # (2*2 - 1*3)*20, negative weights matter
        self.assertEqual(rates[1], 80)  # second scale weight 2, same filter, pixel 1
        self.assertEqual(rates[4], 0)   # negative direction sum rectified only at end
        self.assertEqual(rates[2], 0)

    def test_config_and_conversion_end_to_end(self):
        config = Path(__file__).resolve().parents[1]/'configs/v1_projection.json'
        for name in ['motionProj1', 'motionProj2', 'motionProj3']:
            self.assertEqual(len(load_projection(config, name).weights), 28)
        with tempfile.TemporaryDirectory() as tmp:
            path, out = Path(tmp)/'spikes.dat', Path(tmp)/'converted'
            fixture(path, [(0, 0), (49, 1), (50, 0)])
            result = convert_spikes(path, out, config, 'motionProj1', 101, 50,
                                    dict(fx_px=2, fy_px=2, cx_px=.5, cy_px=.5))
            spec, frames, meta = read_recording(out)
            self.assertEqual(result['bins'], 3)
            self.assertEqual(spec.frame_size, 32)
            self.assertEqual([f.timestamp_s for f in frames], [0, .05, .1])
            self.assertEqual(sum(frames[-1].rates_hz), 0)
            self.assertEqual(meta['provenance']['last_frame_duration_s'], .001)

    def test_stream_failure_rolls_back_only_new_output(self):
        config = Path(__file__).resolve().parents[1]/'configs/v1_projection.json'
        with tempfile.TemporaryDirectory() as tmp:
            path, out = Path(tmp)/'spikes.dat', Path(tmp)/'converted'
            fixture(path, [(0, 0), (150, 1)])
            with self.assertRaises(ValueError):
                convert_spikes(path, out, config, 'motionProj1', 100, 50,
                                dict(fx_px=2, fy_px=2, cx_px=.5, cy_px=.5))
            self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
