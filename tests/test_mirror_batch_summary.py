"""Ensure batch reporting keeps rejected/wrong frames in unconditional errors."""
import json
from contextlib import redirect_stdout
import io
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import run_mstd_mirror_controls as batch_runner


class BatchSummaryTests(unittest.TestCase):
    def test_plan_can_validate_copied_bundle_without_writing_missing_registry_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'configs').mkdir(); (root/'input').mkdir()
            shutil.copyfile(ROOT/'configs/model_python_spiking_geometry_v01.json',root/'configs/model_python_spiking_geometry_v01.json')
            folder=root/batch_runner.SUITE;folder.mkdir(parents=True)
            for name in ('manifest.json','intrinsics.json','cloud_heading_left7.dat','cloud_heading_right7.dat'):
                shutil.copyfile(ROOT/batch_runner.SUITE/name,folder/name)
            registry=root/'input/registry.csv'
            before='path,width,height,frames,channels,sha256,scope\n'
            registry.write_text(before)
            output=io.StringIO()
            args=['run_mstd_mirror_controls.py','--plan-only','--frozen-config',str(ROOT/batch_runner.FREEZE)]
            with patch.object(batch_runner,'ROOT',root),patch.object(sys,'argv',args),redirect_stdout(output):
                batch_runner.main()
            plan=json.loads(output.getvalue())
            self.assertEqual(plan['runs'],6)
            self.assertEqual(plan['input_registry_rows_to_add'],2)
            self.assertEqual(registry.read_text(),before)
            self.assertFalse((root/'experiments').exists())

    def test_rejected_wrong_case_still_contributes_to_error_and_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);batch=root/'batch';batch.mkdir();runs=[]
            for seed in (43,44,45):
                for case in batch_runner.CASES:
                    run_id=f'{case}_{seed}';relative='experiments/validation/'+run_id
                    folder=root/relative;folder.mkdir(parents=True)
                    rejected=(seed==43 and case.endswith('right7'))
                    winner=3 if case.endswith('left7') else 6 if rejected else 5
                    state=dict(winner_index=winner,bearing_valid=not rejected)
                    (folder/'model_config.json').write_text(json.dumps(dict(spiking=dict(seed=seed))))
                    (folder/'model_report.json').write_text(json.dumps(dict(complete=True,frames=80)))
                    (folder/'estimates.jsonl').write_text('\n'.join(json.dumps(state) for _ in range(80)))
                    check=root/(relative+'__gate_check');check.mkdir()
                    (check/'metrics.json').write_text(json.dumps(dict(summary=[dict(
                        unconditional_winner_mae_deg=3 if rejected else 2,diagnostic_passed=not rejected)])))
                    runs.append(dict(case=case,seed=seed,run_id=run_id,output=relative))
            with patch.object(batch_runner,'ROOT',root):batch_runner.summarize_runs(batch,runs)
            result=json.loads((batch/'summary.json').read_text())
            bad=next(r for r in result['cases'] if r['seed']==43 and r['case'].endswith('right7'))
            self.assertEqual(bad['valid_fraction'],0)
            self.assertEqual(bad['mae_deg'],3)
            self.assertFalse(result['all_six_targets_passed'])
            self.assertEqual(result['pairs'][0]['mean_common_bias_deg'],2.5)
            self.assertEqual(result['pairs'][1]['mean_common_bias_deg'],0)


if __name__=='__main__':unittest.main()
