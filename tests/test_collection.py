"""Test collection orchestration with a fake runner, not a biological V1."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

from mstd.calibration import generate_suite
from mstd.carlsim import SIGNATURE


class CollectionTests(unittest.TestCase):
    def load_script(self):
        path = Path(__file__).resolve().parents[1]/'examples/collect_calibration.py'
        spec = importlib.util.spec_from_file_location('collect_calibration',path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module,path

    def test_unique_paths_and_updated_manifest(self):
        module,script = self.load_script()
        with tempfile.TemporaryDirectory() as tmp:
            suite = Path(tmp)/'suite'
            generate_suite(suite,width=4,height=4,frames=2,warmup_frames=1)
            outputs = []
            def fake_run(command,*,cwd,env,stdout,stderr,check):
                self.assertTrue(check)
                self.assertTrue(Path(env['V1_CALIBRATION_INPUT']).is_file())
                output = Path(env['V1_CALIBRATION_OUTPUT'])
                outputs.append(output)
                output.write_bytes(struct.pack('<ifiii',SIGNATURE,.2,4,4,84)+struct.pack('<ii',50,0))
            args = [str(script),'--suite',str(suite),'--executable',str(script),'--workdir',tmp]
            with patch.object(sys,'argv',args),patch.object(module.subprocess,'run',side_effect=fake_run):
                with contextlib.redirect_stdout(io.StringIO()):
                    module.main()
            self.assertEqual(len(set(outputs)),8)
            meta = json.loads((suite/'manifest.json').read_text())
            for run in meta['runs']:
                self.assertTrue((suite/run['spikes']).is_file())
                self.assertEqual(run['acquisition']['spikes_sha256'],module.sha256(suite/run['spikes']))
            with patch.object(sys,'argv',args),contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    module.main()

    def test_ignored_environment_is_not_silently_accepted(self):
        module,script = self.load_script()
        with tempfile.TemporaryDirectory() as tmp:
            suite = Path(tmp)/'suite'
            generate_suite(suite,width=4,height=4,frames=2,warmup_frames=1)
            args = [str(script),'--suite',str(suite),'--executable',str(script),'--workdir',tmp]
            with patch.object(sys,'argv',args),patch.object(module.subprocess,'run'):
                with self.assertRaisesRegex(RuntimeError,'No requested spike file'):
                    module.main()
            meta = json.loads((suite/'manifest.json').read_text())
            self.assertTrue(all(run['spikes'] is None for run in meta['runs']))

    def test_selfmotion_manifest_and_named_subset(self):
        module,script = self.load_script()
        with tempfile.TemporaryDirectory() as tmp:
            suite = Path(tmp)/'suite'
            generate_suite(suite,width=4,height=4,frames=2,warmup_frames=1)
            manifest = suite/'manifest.json'
            meta = json.loads(manifest.read_text())
            meta['schema'] = 'selfmotion-stimulus-v1'
            manifest.write_text(json.dumps(meta))
            def fake_run(command,*,cwd,env,stdout,stderr,check):
                Path(env['V1_CALIBRATION_OUTPUT']).write_bytes(struct.pack('<ifiii',SIGNATURE,.2,4,4,84))
            args = [str(script),'--suite',str(suite),'--executable',str(script),
                    '--workdir',tmp,'--runs','grating_d090']
            with patch.object(sys,'argv',args),patch.object(module.subprocess,'run',side_effect=fake_run) as runner:
                with contextlib.redirect_stdout(io.StringIO()):
                    module.main()
            self.assertEqual(runner.call_count,1)
            meta = json.loads(manifest.read_text())
            self.assertEqual([r['name'] for r in meta['runs'] if r['spikes']],['grating_d090'])


if __name__ == '__main__':
    unittest.main()
