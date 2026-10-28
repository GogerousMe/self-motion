import math
import unittest

from mstd.data import V1Frame
from mstd.mt import encode_mt
from mstd.synthetic import default_spec


class MTChannelTests(unittest.TestCase):
    def test_channels_reconstruct_vectors_without_changing_existing_outputs(self):
        spec = default_spec()
        frame = V1Frame(0, tuple(float((i*17+3)%101) for i in range(spec.frame_size)))
        original = encode_mt(spec, frame)
        exported = encode_mt(spec, frame, return_channels=True)
        self.assertIsNone(original.direction_activity)
        self.assertIsNone(original.pooled_rates_hz)
        self.assertEqual(original.vx, exported.vx)
        self.assertEqual(original.vy, exported.vy)
        self.assertEqual(original.evidence, exported.evidence)
        self.assertEqual(len(exported.pooled_rates_hz), 8)
        channels = exported.direction_activity
        for p in range(spec.width*spec.height):
            self.assertTrue(all(0 <= ch[p] <= 1 for ch in channels))
            self.assertLessEqual(sum(ch[p] for ch in channels), 1.)
            self.assertAlmostEqual(sum(ch[p]*math.cos(a) for ch,a in zip(channels,spec.directions_rad)), exported.vx[p])
            self.assertAlmostEqual(sum(ch[p]*math.sin(a) for ch,a in zip(channels,spec.directions_rad)), exported.vy[p])

    def test_silent_and_single_direction_amplitude(self):
        spec = default_spec()
        n = spec.width*spec.height
        blank = encode_mt(spec, V1Frame(0, (0.,)*spec.frame_size), return_channels=True)
        self.assertTrue(all(v == 0 for ch in blank.direction_activity for v in ch))
        channels = encode_mt(spec, V1Frame(0, (20.,)*n+(0.,)*(spec.frame_size-n)), return_channels=True).direction_activity
        self.assertTrue(all(abs(v-20/21) < 1e-12 for v in channels[0]))
        self.assertTrue(all(v == 0 for ch in channels[1:] for v in ch))
