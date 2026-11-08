"""Numerical/state/encoding regressions for the optional V1 frontend."""
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

HAS_NUMPY = importlib.util.find_spec('numpy') is not None
if HAS_NUMPY:
    import numpy as np
    from mstd.calibration import stimulus_frames,write_stimulus
    from mstd.data import V1Frame,V1Spec
    from mstd.io import read_recording
    from mstd.mt import encode_mt
    from mstd.mstd import MSTd,horizontal_centers
    from mstd.projection import load_projection,project_counts
    from mstd.tensor_pipeline import DirectionMT,TensorMSTd
    from mstd.v1 import ArrayOps,DIFF,GAUSS,MotionEnergyV1,ORIENTATIONS,edge_maps
    from mstd.v1_recording import RawWriter,iter_raw,raw_info
    from mstd.v1_workflow import compare_raw,run_frontend

ROOT = Path(__file__).resolve().parents[1]
PROJECTION = ROOT/'configs/v1_projection.json'


@unittest.skipUnless(HAS_NUMPY,'optional V1 tests require NumPy')
class PythonV1Tests(unittest.TestCase):
    def test_source_correlation_sign_and_zero_padding(self):
        op = ArrayOps()
        value = np.array([1,2,4,8],np.float32)
        np.testing.assert_array_equal(op.correlate(value,DIFF[1],0),[1,1.5,3,-2])
        # Asymmetric derivative must not get reversed by a convolution API.
        impulse = np.zeros((2,5,7),np.float32); impulse[:,2,3]=1
        actual = op.correlate(impulse,DIFF[3],-1)
        np.testing.assert_array_equal(actual[0,2],[0,.5,-1,0,1,-.5,0])

    def test_analytic_impulse_linear_response(self):
        # Independent closed-form separable-kernel calculation at image center.
        # An impulse is written at the NEWEST time slot; the response is evaluated at slot 4.
        m = MotionEnergyV1(25,25)
        frame = np.zeros((25,25),np.uint8); frame[12,12]=255
        stage = m.step(frame,0,True)
        g = np.asarray(GAUSS,np.float64)
        dirs = np.asarray(ORIENTATIONS,np.float64)
        def coeff(order,delta):
            kernel = [1] if order==0 else DIFF[order]
            radius = len(kernel)//2
            return sum(float(k)*g[delta-(i-radius)+4]
                       for i,k in enumerate(kernel) if 0<=delta-(i-radius)+4<9)
        expected = np.zeros(28)
        for ot in range(4):
            for oy in range(4-ot):
                ox = 3-oy-ot
                factor = math.factorial(3)/(math.factorial(ox)*math.factorial(oy)*math.factorial(ot))
                expected += factor*(dirs[0]**ox)*(dirs[1]**oy)*(dirs[2]**ot)*coeff(ox,0)*coeff(oy,0)*coeff(ot,4)
        np.testing.assert_allclose(stage['linear'][0,:,12,12],expected,atol=2e-9,rtol=2e-5)

    def test_history_reset_cadence_rgb_and_profile(self):
        m = MotionEnergyV1(19,21)
        red = np.zeros((21,19,3),np.uint8);red[...,0]=255
        a = m.step(red,0)
        np.testing.assert_allclose(m.history[-1],.1299,atol=2e-8)
        self.assertTrue(np.isfinite(a).all())
        self.assertTrue((a>=0).all())
        with self.assertRaises(ValueError): m.step(red,.08)
        self.assertEqual(m.frame_index,1)
        m.step(red,.05)
        m.reset()
        np.testing.assert_array_equal(m.step(red,0),a)
        with self.assertRaises(ValueError): m.step(red.astype(np.float32),.05)
        m.reset()
        for i in range(10):
            silence = m.step(np.zeros((21,19),np.uint8),i*.05)
            np.testing.assert_array_equal(silence,0)
        self.assertFalse(m.provenance()['scale_smoothing'])

    def test_scale_base_is_shared_but_edges_and_compensation_are_scale_specific(self):
        m = MotionEnergyV1(25,27)
        image = np.random.default_rng(5).integers(0,256,(27,25),dtype=np.uint8)
        stage = m.step(image,0,True)
        edge = edge_maps(25,27)
        for s in (1,2):
            np.testing.assert_allclose(stage['linear'][s,:,8:-8,8:-8],stage['linear'][0,:,8:-8,8:-8],atol=0)
        np.testing.assert_array_equal(stage['rates'][...,[0,-1],:],0)
        np.testing.assert_array_equal(stage['rates'][...,:,[0,-1]],0)
        self.assertGreater(float(m.compensation[0,0,0]),1)
        np.testing.assert_allclose(stage['rates'],stage['before_normalization']/
                                   (stage['normalization_population'][:,None].astype(np.float64)+.01),rtol=1e-6)

    def test_tensor_projection_mt_and_mstd_match_existing_reference(self):
        projection = load_projection(PROJECTION)
        spec = V1Spec(7,6,projection.directions_rad,7,7,3,2.5)
        rates = np.random.default_rng(8).uniform(0,30,(3,28,6,7)).astype(np.float32)
        mt = DirectionMT(ArrayOps(),projection,7,6).step(rates)
        ref_projected = project_counts(spec,(rates*.05).reshape(-1),50,projection)
        np.testing.assert_allclose(mt['direction_rates'].reshape(-1),ref_projected,rtol=2e-5,atol=3e-5)
        reference = encode_mt(spec,V1Frame(0,tuple(ref_projected)),return_channels=True)
        np.testing.assert_allclose(mt['pooled_rates'].reshape(8,-1),reference.pooled_rates_hz,atol=3e-5,rtol=2e-5)
        np.testing.assert_allclose(mt['activity'].reshape(8,-1),reference.direction_activity,atol=1e-6)
        np.testing.assert_allclose(mt['vectors'].reshape(2,-1),[reference.vx,reference.vy],atol=1e-6)
        self.assertAlmostEqual(mt['evidence'],reference.evidence,places=6)
        centers = horizontal_centers(spec,[-10,0,10])
        a,b = MSTd(spec,centers),TensorMSTd(ArrayOps(),spec,centers)
        for _ in range(4):
            ra,rb = a.step(reference,.05),b.step(mt,.05)
            np.testing.assert_allclose(rb['activity'],ra['activity'],atol=2e-6)
            np.testing.assert_allclose(rb['template_scores'],ra['template_scores'],atol=2e-6)
            self.assertEqual(ra['winner_index'],rb['winner_index'])

    def test_offline_outputs_existing_protocol_and_strict_pair_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            stim=root/'right.dat'
            write_stimulus(stim,19,19,stimulus_frames(19,19,4,0,1.5))
            camera=dict(fx_px=19,fy_px=19,cx_px=9,cy_px=9)
            report=run_frontend(stim,root/'run',PROJECTION,max_frames=3,save_raw=True,save_mt=True,
                                warmup_frames=1,intrinsics=camera,progress_every=0)
            self.assertEqual(report['frames'],3)
            self.assertTrue(report['mstd_enabled'])
            spec,frames,meta=read_recording(root/'run')
            self.assertEqual(len(frames),3)
            self.assertFalse(meta['provenance']['complete_stimulus'])
            raw=root/'run/raw_v1'
            self.assertEqual(raw_info(raw)['frames'],3)
            result=compare_raw(raw,raw,root/'same',PROJECTION,atol=0,rtol=0)
            self.assertTrue(result['passed'])
            changed=root/'changed'
            original=raw_info(raw)
            writer=RawWriter(changed,19,19,.05,original['provenance'])
            for i,a in enumerate(iter_raw(raw)):
                a=a.copy()
                if i==1: a[0,0,9,9]+=1
                writer.append(a)
            writer.close()
            result=compare_raw(raw,changed,root/'different',PROJECTION,atol=0,rtol=0)
            self.assertFalse(result['passed'])
            original['provenance']['stimulus_sha256']='wrong'
            (changed/'metadata.json').write_text(json.dumps(original),encoding='utf-8')
            with self.assertRaises(ValueError): compare_raw(raw,changed,root/'mispaired',PROJECTION)
            self.assertFalse((root/'mispaired').exists())
            with (raw/'rates.f32').open('ab') as stream:stream.write(b'bad')
            with self.assertRaises(ValueError):raw_info(raw)

    @unittest.skipUnless(importlib.util.find_spec('torch'),'optional PyTorch not installed')
    def test_torch_cpu_and_optional_cuda_match_numpy(self):
        import torch
        torch.set_num_threads(1)
        devices=['cpu']+(['cuda'] if torch.cuda.is_available() else [])
        projection=load_projection(PROJECTION)
        images=[np.random.default_rng(i).integers(0,256,(21,19),dtype=np.uint8) for i in range(4)]
        for device in devices:
            a,b=MotionEnergyV1(19,21),MotionEnergyV1(19,21,backend='torch',device=device)
            ma,mb=DirectionMT(a.ops,projection,19,21),DirectionMT(b.ops,projection,19,21)
            for i,image in enumerate(images):
                ra,rb=a.step(image,i*.05),b.step(image,i*.05)
                np.testing.assert_allclose(b.ops.host(rb),ra,rtol=5e-4,atol=5e-4)
                with torch.inference_mode():
                    ta,tb=ma.step(ra),mb.step(rb)
                np.testing.assert_allclose(b.ops.host(tb['activity']),ta['activity'],atol=5e-4,rtol=5e-4)


if __name__=='__main__':unittest.main()
