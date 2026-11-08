"""Saved-spike viewer validates units/timing and reads genuine sparse events."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from mstd.calibration import write_stimulus,stimulus_frames
from mstd.simulation import simulate

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('spiking_view',ROOT/'examples/visualize_python_spiking.py')
VIEW=importlib.util.module_from_spec(spec);spec.loader.exec_module(VIEW)


class SpikingVisualizationTests(unittest.TestCase):
    def fixture(self,root):
        stimulus=root/'input.dat';write_stimulus(stimulus,19,21,stimulus_frames(19,21,3,0,1.5))
        output=root/'run'
        simulate(dict(mt_backend='python_spiking',mt_radius=0,input=dict(path=str(stimulus)),
                      projection_config=str(ROOT/'configs/v1_projection.json'),output=str(output),progress_every=0,
                      monitors=dict(spikes=True,neural_state=True)))
        return stimulus,output

    def test_frame_end_metadata_pairing_and_observed_event_times(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);stimulus,output=self.fixture(root)
            data=VIEW.load_recording(output)
            self.assertEqual(data['indices'],[0,1,2])
            self.assertEqual(list(data['order']),[0,7,6,5,4,3,2,1])
            self.assertEqual(VIEW.paired_stimulus(data)[0],stimulus)
            time,ids,counts,n,total=VIEW.raster_events(data,'exc',0,0,.15)
            self.assertTrue(np.all(time>0));self.assertTrue(np.all(time<.15))
            self.assertTrue(np.all(counts>0));self.assertTrue(np.all(ids<n))
            bad=root/'wrong.dat';raw=bytearray(stimulus.read_bytes());raw[-1]^=1;bad.write_bytes(raw)
            with self.assertRaisesRegex(ValueError,'SHA256'):VIEW.paired_stimulus(data,bad)
            path=output/'monitors/mt_inh_rates/metadata.json'
            meta=json.loads(path.read_text());meta['timestamps_s']=[0,.05,.1]
            path.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'timing'):VIEW.load_recording(output)
            VIEW.close_recording(data)

    def test_raster_rejects_out_of_duration_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,output=self.fixture(root);data=VIEW.load_recording(output)
            path=output/'monitors/spikes/exc.events'
            rows=np.fromfile(path,dtype=VIEW.EVENT_DTYPE)
            if not len(rows):self.fail('fixture must contain excitatory events')
            rows['tick'][-1]=150;rows.tofile(path)
            with self.assertRaisesRegex(ValueError,'time/id'):VIEW.raster_events(data,'exc',0,0,.15)
            VIEW.close_recording(data)


if __name__=='__main__':unittest.main()
