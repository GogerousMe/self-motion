import importlib.util
from math import pi
from pathlib import Path
import sys
import unittest

from mstd.data import V1Spec
from mstd.projection import Projection

EXAMPLES = Path(__file__).resolve().parents[1]/'examples'
SPEC = importlib.util.spec_from_file_location('compare_mt_stages', EXAMPLES/'compare_mt_stages.py')
COMPARE = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(EXAMPLES))
try:
    SPEC.loader.exec_module(COMPARE)
finally:
    sys.path.pop(0)


class StageComparisonTests(unittest.TestCase):
    def test_local_impulse_remains_local_and_normalizes_per_pixel(self):
        spec = V1Spec(4, 4, tuple(d*pi/4 for d in range(8)), 1, 1, 0, 0)
        weights = tuple(tuple(float(c == d) for d in range(8)) for c in range(28))
        projection = Projection('test', weights, (1., 1., 1.), spec.directions_rad, 1.)
        counts = [0]*(84*16)
        counts[1*16+5] = 1  # direction45, pixel(1,1), one event in 50ms
        projected, pooled, normalized = COMPARE.compute_stages(spec, counts, 50, projection, 1)
        self.assertEqual(projected[1][5], 20.)
        self.assertEqual(sum(projected[1]), 20.)
        self.assertAlmostEqual(pooled[1][5], 20/9)
        self.assertAlmostEqual(pooled[1][0], 20/4)
        self.assertEqual(pooled[1][15], 0.)  # no global illumination / spatial broadcast
        self.assertAlmostEqual(normalized[1][5], (20/9)/(1+20/9))
        self.assertEqual(normalized[1][15], 0.)
        self.assertTrue(all(sum(ch) == 0 for d,ch in enumerate(normalized) if d != 1))

    def test_radius_zero_retains_projected_spatial_structure(self):
        spec = V1Spec(4, 4, tuple(d*pi/4 for d in range(8)), 1, 1, 0, 0)
        weights = tuple(tuple(float(c == d) for d in range(8)) for c in range(28))
        projection = Projection('test', weights, (1., 1., 1.), spec.directions_rad, 1.)
        counts = [0]*(84*16)
        counts[15] = 2
        projected, pooled, normalized = COMPARE.compute_stages(spec, counts, 50, projection, 0)
        self.assertEqual(projected, pooled)
        self.assertAlmostEqual(normalized[0][15], 40/41)
