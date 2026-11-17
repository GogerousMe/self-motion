"""CLI plans preserve spatial controls and expose the four-way experiment."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from run_mt_spatial_controls import VARIANT_SWITCHES, parse_args, run


class CompetitionPlanTests(unittest.TestCase):
    def test_four_way_plan_contains_both_independent_switches(self):
        args = parse_args(['--device', 'cpu', '--quick-check', '--plan-only'], competition=True)
        output = io.StringIO()
        with redirect_stdout(output):
            run(args, variants=tuple(VARIANT_SWITCHES))
        plan = json.loads(output.getvalue())
        self.assertEqual(plan['variants'], ['baseline', 'spatial_only', 'competition_only', 'spatial_competition'])
        self.assertEqual(plan['execution'], dict(backend='numpy', device='cpu'))
        self.assertEqual(len(plan['cases']), 2)
        self.assertEqual(plan['mt_competition']['mode'], 'angular')
        self.assertEqual(plan['spiking']['substeps'], 32)

    def test_existing_spatial_plan_preserves_its_protocol(self):
        args = parse_args(['--device', 'cpu', '--plan-only'])
        output = io.StringIO()
        with redirect_stdout(output):
            run(args)
        plan = json.loads(output.getvalue())
        self.assertEqual(plan['schema'], 'mstd-mt-spatial-controls-v1')
        self.assertEqual(len(plan['cases']), 33)
        self.assertEqual(plan['texture_width'], 80)
        self.assertEqual(plan['texture_frames'], 40)

    def test_invalid_competition_parameters_fail_before_run_creation(self):
        args = parse_args(['--device', 'cpu', '--quick-check', '--plan-only', '--competition-weight', '-1'], competition=True)
        with self.assertRaises(ValueError):
            run(args, variants=tuple(VARIANT_SWITCHES))
        with self.assertRaises(ValueError):
            run(args, variants=('competition_only',))


if __name__ == '__main__':
    unittest.main()
