import json
import math
from pathlib import Path
import tempfile
import unittest

from mstd.data import V1Frame, V1Spec
from mstd.io import read_recording, write_recording
from mstd.mt import encode_mt
from mstd.mstd import DynamicsConfig, MSTd, horizontal_centers
from mstd.synthetic import default_spec, radial_frame


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.spec = default_spec()

    def model(self, config=None):
        return MSTd(self.spec, horizontal_centers(self.spec, range(-20, 21, 5)), config)

    def test_round_trip_channel_layout(self):
        values = tuple(float(i % 79) for i in range(self.spec.frame_size))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'recording'
            write_recording(path, self.spec, [V1Frame(0, values)], {'source': 'test'})
            spec, frames, _ = read_recording(path)
            self.assertEqual(spec, self.spec)
            self.assertEqual(frames[0].rates_hz, values)
            with self.assertRaises(FileExistsError):
                write_recording(path, self.spec, frames, {})

    def test_reject_wrong_size_nonfinite_negative_and_duplicate_time(self):
        for values in [(0.0,), (float('nan'),)*self.spec.frame_size,
                       (-1.0,)*self.spec.frame_size]:
            with self.assertRaises(ValueError):
                V1Frame(0, values).validate(self.spec)
        with tempfile.TemporaryDirectory() as tmp:
            frame = radial_frame(self.spec, 0)
            with self.assertRaises(ValueError):
                write_recording(Path(tmp)/'bad', self.spec, [frame, frame], {})
            self.assertFalse((Path(tmp)/'bad').exists())

    def test_reject_truncated_binary_and_wrong_encoding(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'recording'
            write_recording(path, self.spec, [radial_frame(self.spec, 0)], {})
            binary = path/'rates.f32'
            binary.write_bytes(binary.read_bytes()[:-4])
            with self.assertRaises(ValueError):
                read_recording(path)
            meta = json.loads((path/'metadata.json').read_text())
            meta['encoding'] = 'raw_28x3_filters'
            (path/'metadata.json').write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                read_recording(path)

    def test_heading_sign_and_template_selection(self):
        for expected in [-15, 0, 15]:
            model = self.model()
            field = encode_mt(self.spec, radial_frame(self.spec, 0, expected), radius=0)
            result = model.step(field, 0.5)
            self.assertTrue(result['valid'])
            self.assertAlmostEqual(result['alpha_x_deg'], expected, places=6)

    def test_silence_and_isotropic_activity_have_no_heading(self):
        model = self.model()
        model.step(encode_mt(self.spec, radial_frame(self.spec, 0, 15)), 0.5)
        for rate in [0.0, 30.0]:
            frame = V1Frame(0, (rate,)*self.spec.frame_size)
            result = model.step(encode_mt(self.spec, frame), 0.05)
            self.assertFalse(result['valid'])
            self.assertIsNone(result['alpha_x_deg'])

    def test_dynamics_bounded_and_numerically_convergent(self):
        mt = encode_mt(self.spec, radial_frame(self.spec, 0, -10))
        a = self.model(DynamicsConfig(max_step_s=0.002))
        b = self.model(DynamicsConfig(max_step_s=0.001))
        ra, rb = a.step(mt, 0.2), b.step(mt, 0.2)
        self.assertEqual(ra['winner_index'], rb['winner_index'])
        self.assertLess(max(abs(x-y) for x,y in zip(a.activity,b.activity)), 0.003)
        self.assertTrue(all(math.isfinite(x) and 0 <= x <= 1 for x in a.activity))
        a.reset()
        self.assertEqual(sum(a.activity), 0)

    def test_bad_intrinsics_directions_and_integration_parameters(self):
        with self.assertRaises(ValueError):
            V1Spec(4, 4, (0, 0, 1, 2), 1, 1, 0, 0)
        with self.assertRaises(ValueError):
            V1Spec(4, 4, self.spec.directions_rad, 0, 1, 0, 0)
        with self.assertRaises(ValueError):
            DynamicsConfig(max_step_s=1)
        with self.assertRaises(ValueError):
            self.model().step(encode_mt(self.spec, radial_frame(self.spec, 0)), float('nan'))


if __name__ == '__main__':
    unittest.main()
