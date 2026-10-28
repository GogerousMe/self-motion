import importlib.util
from pathlib import Path
import unittest

from mstd.data import V1Spec
from math import pi

PATH = Path(__file__).resolve().parents[1]/'examples/visualize_mt_gratings.py'
SPEC = importlib.util.spec_from_file_location('visualize_mt_gratings', PATH)
VIDEO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VIDEO)


class RateVisualizationTests(unittest.TestCase):
    def test_counts_to_hz_layout_and_window_duration(self):
        spec = V1Spec(2, 2, tuple(i*pi/4 for i in range(8)), 1, 1, 0, 0)
        counts = [0]*32
        counts[4+2] = 3  # right-down plane, pixel 2
        maps = VIDEO.rate_channels(counts, spec, 50)
        self.assertEqual(maps[1][2], 60.)
        self.assertEqual(sum(sum(ch) for ch in maps), 60.)
        self.assertEqual(VIDEO.rate_channels(counts, spec, 1000)[1][2], 3.)
        with self.assertRaises(ValueError):
            VIDEO.rate_channels(counts[:-1], spec, 50)
        with self.assertRaises(ValueError):
            VIDEO.rate_channels(counts, spec, 0)

    def test_rate_color_scale_clips_and_jet_has_blue_low_red_high(self):
        self.assertEqual(VIDEO.color(200, 100, 'jet'), VIDEO.color(100, 100, 'jet'))
        self.assertGreater(VIDEO.color(0, 100, 'jet')[2], VIDEO.color(0, 100, 'jet')[0])
        self.assertGreater(VIDEO.color(100, 100, 'jet')[0], VIDEO.color(100, 100, 'jet')[2])
