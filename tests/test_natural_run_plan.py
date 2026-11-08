"""Plan and registry semantics only: no expensive natural-video simulation."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from mstd.calibration import write_stimulus,stimulus_frames,file_sha256

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('natural_runs',ROOT/'tools/run_natural_spiking.py')
N=importlib.util.module_from_spec(spec);spec.loader.exec_module(N)


class NaturalRunPlanTests(unittest.TestCase):
    def fixture(self,root):
        sid='daria_walk__19x21__f000003__dt50ms__gray_native__v01'
        path=Path('input/stimuli/natural/daria_walk')/sid/(sid+'.dat')
        (root/path).parent.mkdir(parents=True)
        write_stimulus(root/path,19,21,stimulus_frames(19,21,3,0,1.5))
        digest=file_sha256(root/path)
        (root/path).with_suffix('.dat.json').write_text(json.dumps(dict(frame_ms=50,stimulus_sha256=digest)))
        with (root/'input/registry.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['path','width','height','frames','channels','sha256','scope'])
            w.writeheader();w.writerow(dict(path=path.as_posix(),width=19,height=21,frames=3,channels=1,sha256=digest,scope='fixture'))
        (root/'experiments').mkdir()
        with (root/'experiments/registry.csv').open('w',newline='') as f:csv.writer(f).writerow(N.FIELDS)
        (root/'configs').mkdir()
        (root/'configs/model_python_spiking.json').write_bytes((ROOT/'configs/model_python_spiking.json').read_bytes())
        return sid

    def test_all_frames_and_preview_names_follow_actual_input(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);sid=self.fixture(root)
            p=N.plan(root,'daria_walk',device='cpu',stimulus_id=sid,stamp='20261108_120000')
            self.assertEqual(p['frames'],3);self.assertIsNone(p['config']['max_frames'])
            self.assertEqual(p['run_id'],'20261108_120000__daria_walk_19x21_f000003__py_spike_cpu__r01')
            self.assertTrue(p['recording'].startswith('outputs/recordings/daria_walk/'))
            self.assertEqual(p['frame'],2)
            preview=N.plan(root,'daria_walk',stage='scratch',max_frames=1,repeat=2,stimulus_id=sid,stamp='20261108_120001')
            self.assertEqual((preview['config']['backend'],preview['config']['device']),('torch','cuda'))
            self.assertIn('py_spike_cuda',preview['run_id'])
            self.assertTrue(preview['recording'].startswith('experiments/scratch/'))
            self.assertIn('f000001',preview['run_id']);self.assertTrue(preview['run_id'].endswith('r02'))

    def test_registry_transitions_preserve_history_and_reject_duplicate(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);sid=self.fixture(root)
            p=N.plan(root,'daria_walk',stimulus_id=sid,stamp='20261108_120000')
            N.record(root,p,'recording','planned')
            with self.assertRaises(ValueError):N.record(root,p,'recording','planned')
            N.record(root,p,'recording','running')
            (root/p['recording']).mkdir(parents=True)
            N.record(root,p,'recording','completed')
            artifact=json.loads((root/p['recording']/'artifact.json').read_text())
            self.assertEqual(artifact['status'],'completed');self.assertEqual(artifact['stimulus_ids'],[sid])
            with self.assertRaises(ValueError):N.record(root,p,'recording','running')
            N.record(root,p,'visualization','planned');N.record(root,p,'visualization','failed','missing ffmpeg fixture')
            with (root/'experiments/registry.csv').open(newline='') as f:rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),2);self.assertEqual(rows[1]['parent_run_ids'],p['run_id'])

    def test_hash_mismatch_rejected_before_any_artifact(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);sid=self.fixture(root)
            path=next((root/'input/stimuli').rglob('*.dat'))
            raw=bytearray(path.read_bytes());raw[-1]^=1;path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError,'registry'):N.plan(root,'daria_walk',stimulus_id=sid)
            self.assertFalse((root/'outputs').exists())


if __name__=='__main__':unittest.main()
