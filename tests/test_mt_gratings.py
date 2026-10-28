import csv
import importlib.util
import json
from math import pi
from pathlib import Path
import struct
import tempfile
import unittest

from mstd.calibration import file_sha256, write_stimulus
from mstd.carlsim import SIGNATURE

SCRIPT = Path(__file__).resolve().parents[1]/'examples/analyze_mt_gratings.py'
MODULE_SPEC = importlib.util.spec_from_file_location('mt_gratings', SCRIPT)
MODULE = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(MODULE)


class MTGratingTests(unittest.TestCase):
    def fixture(self, root, reversed_direction=False, silent=False):
        stimulus = root/'all_stimuli.dat'
        write_stimulus(stimulus, 4, 4, [bytes(16)]*32)
        segments = []
        events = []
        for i in range(8):
            start = i*200
            for blank, offset in [(0, 0), (1, 100)]:
                segments.append(dict(segment_index=len(segments)+1, direction_slot_deg=i*45,
                    speed_px_per_frame=0 if blank else 1.5, is_blank=blank,
                    first_frame=(start+offset)//50+1, last_frame=(start+offset+100)//50,
                    start_ms=start+offset, end_ms=start+offset+100))
            channel = (-i + (4 if reversed_direction else 0)) % 8
            if not silent:
                # Deliberately wrong during warmup; exact boundary starts valid bin.
                for t, c in [(start, (channel+4)%8), (start+50, channel)]:
                    events.extend((t, c*16+p) for p in range(16))
        path = root/'segments.csv'
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(segments[0]))
            writer.writeheader()
            writer.writerows(segments)
        manifest = root/'manifest.json'
        manifest.write_text(json.dumps(dict(schema='matlab-sequential-binary-gratings-v2',
            coordinates='x-right_y-down; clockwise from right', width=4, height=4, frame_ms=50,
            runs=dict(duration_ms=1600, frames=32, stimulus=stimulus.name,
                      stimulus_sha256=file_sha256(stimulus), segments=path.name))))
        spikes = root/'spikes.dat'
        spikes.write_bytes(struct.pack('<ifiii', SIGNATURE, .2, 4, 4, 84)
                           + b''.join(struct.pack('<ii', *e) for e in events))
        config = root/'projection.json'
        config.write_text(json.dumps(dict(schema='v1-filter-projection-v1', gain=1,
            scale_weights=[1, 1, 1], directions_rad=[-i*pi/4 for i in range(8)],
            projections={'test': {'weights': [[float(i == j) for j in range(8)] for i in range(28)]}})))
        return manifest, spikes, config

    def test_actual_mt_eight_directions_warmup_boundaries_and_blank(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, spikes, config = self.fixture(root)
            report = MODULE.analyze(manifest, spikes, root/'out', config, 'test', 50, 1)
            self.assertTrue(report['moving_directions_passed'])
            for row in report['segments']:
                if row['is_blank']:
                    self.assertIsNone(row['expected_direction_deg'])
                    self.assertIsNone(row['within_tolerance'])
                    self.assertFalse(row['responsive'])
                else:
                    self.assertLess(row['mt_error_deg'], 1e-9)
                    self.assertGreater(row['mt_evidence'], .9)
            with (root/'out/frames.csv').open() as stream:
                frames = list(csv.DictReader(stream))
            self.assertEqual(len(frames), 32)
            self.assertEqual(frames[0]['evaluation_bin'], 'False')
            self.assertEqual(frames[1]['evaluation_bin'], 'True')
            with self.assertRaises(FileExistsError):
                MODULE.analyze(manifest, spikes, root/'out', config, 'test', 50, 1)

    def test_reversed_silent_and_no_evaluation_are_not_passed(self):
        for reversed_direction, silent, warmup in [(True, False, 50), (False, True, 50), (False, False, 100)]:
            with self.subTest(reversed=reversed_direction, silent=silent, warmup=warmup):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    manifest, spikes, config = self.fixture(root, reversed_direction, silent)
                    report = MODULE.analyze(manifest, spikes, root/'out', config, 'test', warmup, 1)
                    self.assertFalse(report['moving_directions_passed'])

    def test_reject_checksum_and_incomplete_spike_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, spikes, config = self.fixture(root)
            with spikes.open('ab') as stream:
                stream.write(struct.pack('<ii', 1600, 0))
            with self.assertRaisesRegex(ValueError, 'duration'):
                MODULE.analyze(manifest, spikes, root/'bad_time', config, 'test', 50, 1)
            self.assertFalse((root/'bad_time/report.json').exists())
            stimulus = root/'all_stimuli.dat'
            data = bytearray(stimulus.read_bytes())
            data[-1] ^= 1
            stimulus.write_bytes(data)
            with self.assertRaisesRegex(ValueError, 'checksum'):
                MODULE.analyze(manifest, spikes, root/'bad_hash', config, 'test', 50, 1)


if __name__ == '__main__':
    unittest.main()
