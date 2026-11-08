import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('mt_compare', Path(__file__).parents[1]/'examples/compare_carlsim_python_mt.py')
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


class ComparisonTests(unittest.TestCase):
    def test_half_open_bins_ids_and_explicit_silent_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'spikes.dat'
            path.write_bytes(struct.pack('<ifiii', 206661989, .2, 2, 3, 8)
                             + b''.join(struct.pack('<ii', *e) for e in [(0,0),(49,0),(50,47)]))
            rates, meta = compare.bin_spikes(path, 50, 150)
            self.assertEqual(rates.shape, (3,8,3,2))
            self.assertEqual(rates[0,0,0,0], 40)
            self.assertEqual(rates[1,7,2,1], 20)
            self.assertEqual(rates[2].sum(), 0)
            self.assertFalse(meta['duration_inferred'])
            inferred, meta = compare.bin_spikes(path, 50)
            self.assertEqual(len(inferred), 2)
            self.assertTrue(meta['duration_inferred'])
            with self.assertRaises(ValueError):
                compare.bin_spikes(path, 50, 50)

    def test_direction_mapping_and_silent_profiles(self):
        self.assertEqual(compare.direction_order([0,315,270,225,180,135,90,45]).tolist(), [0,7,6,5,4,3,2,1])
        with self.assertRaises(ValueError):
            compare.direction_order([0]*8)
        self.assertIsNone(compare.profile(np.zeros(8)))
        p = compare.profile(np.array([0,0,0,0,10,0,0,0]))
        self.assertEqual(p['winner_deg'], 180)
        self.assertAlmostEqual(p['vector_deg'], 180)
        self.assertAlmostEqual(p['resultant'], 1)


if __name__ == '__main__':
    unittest.main()
