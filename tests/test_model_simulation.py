"""Unified layers and passive-monitor behavior, using the same existing model equations."""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

HAS_NUMPY=importlib.util.find_spec('numpy') is not None
if HAS_NUMPY:
    import numpy as np
    from mstd.calibration import stimulus_frames,write_stimulus
    from mstd.model import VisionModel,model_config
    from mstd.simulation import simulate
    from mstd.v1_recording import iter_raw,iter_stimulus,raw_info

PROJECTION=str(Path(__file__).resolve().parents[1]/'configs/v1_projection.json')


@unittest.skipUnless(HAS_NUMPY,'unified model tests require NumPy')
class ModelSimulationTests(unittest.TestCase):
    def settings(self,**extra):
        return model_config({'projection_config':PROJECTION,'progress_every':0,**extra})

    def test_all_visual_stimulus_categories_preserve_pixels_and_model_responses(self):
        # Original CARLsim reads identical byte frames for categories 0..6.
        # Include non-square RGB so a category fix cannot hide channel/axis errors.
        images=np.random.default_rng(31).integers(0,256,(3,21,19,3),dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            reference=None
            for kind in (4,0,1,2,3,5,6):
                with self.subTest(kind=kind):
                    stim=root/f'type_{kind}.dat'
                    stim.write_bytes(struct.pack('<ifiBiii',293390619,1.,kind,3,19,21,3)+images.tobytes())
                    for index,image in iter_stimulus(stim):
                        np.testing.assert_array_equal(image,images[index])
                    output=root/f'run_{kind}'
                    report=simulate(self.settings(input={'path':str(stim)},output=str(output)))
                    self.assertEqual(report['frames'],3)
                    arrays={stage:np.fromfile(output/'monitors'/stage/name,dtype='<f4')
                            for stage,name in [('v1','rates.f32'),('mt_projection','values.f32'),
                                ('mt_pooling','values.f32'),('mt_normalization','values.f32'),
                                ('mstd_activity','values.f32')]}
                    if reference is None:
                        reference=(report['last_mstd'],arrays)
                    else:
                        self.assertEqual(report['last_mstd'],reference[0])
                        for stage,values in arrays.items():
                            np.testing.assert_array_equal(values,reference[1][stage])
            for kind in (-1,7):
                stim=root/f'invalid_{kind}.dat'
                stim.write_bytes(struct.pack('<ifiBiii',293390619,1.,kind,3,19,21,3)+images.tobytes())
                output=root/f'bad_{kind}'
                with self.assertRaises(ValueError):
                    simulate(self.settings(input={'path':str(stim)},output=str(output)))
                self.assertFalse(output.exists())
            stim=root/'type_6.dat'
            stim.write_bytes(stim.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError,'size mismatch'):
                simulate(self.settings(input={'path':str(stim)},output=str(root/'truncated')))
            self.assertFalse((root/'truncated').exists())

    def test_layer_dependency_checks_and_pixel_space_mstd(self):
        with self.assertRaises(ValueError):self.settings(layers={'v1':False})
        with self.assertRaises(ValueError):self.settings(layers={'mt':False})
        with self.assertRaises(ValueError):self.settings(monitors={'every_n_frames':0})
        with self.assertRaises(ValueError):self.settings(layers={'v1':'false'})
        config=self.settings(layers={'mt':False,'mstd':False})
        model=VisionModel(19,21,config)
        image=np.random.default_rng(1).integers(0,256,(21,19),dtype=np.uint8)
        state=model.step(image,0)
        self.assertIsNone(state['mt']);self.assertIsNone(state['mstd'])
        full=VisionModel(19,21,self.settings())
        state=full.step(image,0)
        self.assertEqual(len(state['mstd']['activity']),9)
        self.assertFalse(state['mstd']['bearing_valid'])
        self.assertIsNone(state['mstd']['alpha_x_deg'])
        self.assertFalse(full.description()['bearing_calibrated'])
        full.reset()
        again=full.step(image,0)
        np.testing.assert_array_equal(state['v1'],again['v1'])
        np.testing.assert_array_equal(state['mstd']['activity'],again['mstd']['activity'])

    def test_monitors_do_not_change_computation_and_stride_only_changes_storage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);stim=root/'stim.dat'
            write_stimulus(stim,19,19,stimulus_frames(19,19,5,0,1.5))
            base={'input':{'path':str(stim)},'output':str(root/'all')}
            first=simulate(self.settings(**base))
            silent={k:False for k in ('v1','mt_projection','mt_pooling','mt_normalization','mstd','summary')}
            second=simulate(self.settings(input={'path':str(stim)},output=str(root/'none'),monitors=silent))
            third=simulate(self.settings(input={'path':str(stim)},output=str(root/'sparse'),
                                          monitors={'every_n_frames':2}))
            self.assertEqual(first['frames'],5)
            self.assertEqual(first['last_mstd'],second['last_mstd'])
            self.assertEqual(first['last_mstd'],third['last_mstd'])
            self.assertFalse((root/'none/monitors').exists())
            self.assertFalse((root/'none/frames.csv').exists())
            raw=raw_info(root/'all/monitors/v1')
            sparse=raw_info(root/'sparse/monitors/v1')
            self.assertEqual(raw['frames'],5)
            self.assertEqual(sparse['sample_indices'],[0,2,4])
            values=list(iter_raw(root/'all/monitors/v1'))
            samples=list(iter_raw(root/'sparse/monitors/v1'))
            for a,b in zip(values[::2],samples):np.testing.assert_array_equal(a,b)
            for stage in ('mt_projection','mt_pooling','mt_normalization','mstd_activity','mstd_scores'):
                meta=json.loads((root/'all/monitors'/stage/'metadata.json').read_text())
                self.assertTrue(meta['complete'])
                self.assertEqual(meta['shape'][0],5)
                self.assertEqual((root/'all/monitors'/stage/'values.f32').stat().st_size,
                                 int(np.prod(meta['shape']))*4)

    def test_v1_replay_connects_mt_mstd_and_calibrated_readout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);stim=root/'stim.dat'
            write_stimulus(stim,19,19,stimulus_frames(19,19,4,0,1.5))
            camera={'fx_px':19,'fy_px':19,'cx_px':9,'cy_px':9}
            first=simulate(self.settings(input={'path':str(stim)},output=str(root/'original'),intrinsics=camera))
            replay=simulate(self.settings(input={'kind':'v1_recording','path':str(root/'original/monitors/v1')},
                                          layers={'v1':False},output=str(root/'replay'),intrinsics=camera))
            self.assertTrue(first['model']['bearing_calibrated'])
            self.assertEqual(first['last_mstd'],replay['last_mstd'])
            a=np.fromfile(root/'original/monitors/mt_normalization/values.f32',dtype='<f4')
            b=np.fromfile(root/'replay/monitors/mt_normalization/values.f32',dtype='<f4')
            np.testing.assert_array_equal(a,b)
            sparse=simulate(self.settings(input={'path':str(stim)},output=str(root/'sparse'),
                                           monitors={'every_n_frames':2}))
            with self.assertRaises(ValueError):
                simulate(self.settings(input={'kind':'v1_recording','path':str(root/'sparse/monitors/v1')},
                                       layers={'v1':False},output=str(root/'bad')))
            self.assertFalse((root/'bad').exists())


if __name__=='__main__':unittest.main()
