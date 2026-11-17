"""Config, timestep, causal state and fixture tests for MSTd dynamics."""
from dataclasses import asdict
import importlib.util
from pathlib import Path
import unittest

import numpy as np

from mstd.data import V1Spec
from mstd.model import VisionModel,model_config
from mstd.mstd import DynamicsConfig,horizontal_centers,MSTd
from mstd.mt import MTField
from mstd.readout import ReadoutConfig
from mstd.temporal import temporal_variants,ideal_sequence,response_metrics
from mstd.tensor_pipeline import TensorMSTd
from mstd.v1 import ArrayOps
from mstd.validity import ValidityConfig

SPEC=V1Spec(16,16,tuple(np.arange(8)*np.pi/4),16,16,7.5,7.5)
PROJECTION=str(Path(__file__).resolve().parents[1]/'configs/v1_projection.json')


class TemporalTests(unittest.TestCase):
    def test_config_reaches_normal_model_and_defaults_reproduce(self):
        settings=dict(backend='numpy',device='cpu',mt_backend='python_spiking',mt_radius=0,
                      projection_config=PROJECTION,frame_ms=1,input=dict(kind='v1_recording'),layers=dict(v1=False))
        a=VisionModel(2,2,settings)
        b=VisionModel(2,2,{**settings,'mstd_dynamics':asdict(DynamicsConfig())})
        raw=np.full((3,28,2,2),20,np.float32)
        for i in range(10):
            sa,sb=a.step(raw,i*.001),b.step(raw,i*.001)
            self.assertEqual(sa['mstd'],sb['mstd'])
        c=VisionModel(2,2,{**settings,'mstd_dynamics':dict(recurrent_gain=0,tau_s=.16)})
        self.assertEqual(c.mstd.config.recurrent_gain,0)
        self.assertEqual(c.description()['mstd_dynamics']['tau_s'],.16)
        for bad in ({'tau_s':0},{'gain':True},{'inhibition':'1'},{'max_step_s':float('nan')},{'tau_s':.001},{'other':1}):
            with self.assertRaises(ValueError):model_config({**settings,'mstd_dynamics':bad})
        self.assertEqual(DynamicsConfig(tau_s=np.float32(.08)).tau_s,np.float32(.08))

    def test_variants_state_partition_reset_and_default_identity(self):
        centers=horizontal_centers(SPEC,np.linspace(-20,20,9))
        vectors,_=next(ideal_sequence(SPEC,'hold',80))
        signal=dict(vectors=vectors,evidence=.7)
        self.assertEqual(temporal_variants()['reference'],DynamicsConfig())
        for config in temporal_variants().values():
            a=TensorMSTd(ArrayOps(),SPEC,centers,config)
            b=TensorMSTd(ArrayOps(),SPEC,centers,config)
            a.step(signal,.05)
            for _ in range(10):b.step(signal,.005)
            np.testing.assert_array_equal(a.activity,b.activity)
            a.reset();np.testing.assert_array_equal(a.activity,np.zeros(9))

    def test_list_and_numpy_equations_match_with_ablations(self):
        centers=horizontal_centers(SPEC,np.linspace(-20,20,9))
        v,_=next(ideal_sequence(SPEC,'hold',80))
        field=MTField(tuple(v[0].ravel()),tuple(v[1].ravel()),.7)
        for cfg in temporal_variants().values():
            a=MSTd(SPEC,centers,cfg)
            b=TensorMSTd(ArrayOps(),SPEC,centers,cfg)
            for _ in range(40):
                a.step(field,.001);b.step(dict(vectors=v,evidence=.7),.001)
            np.testing.assert_allclose(a.activity,b.activity,atol=2e-6)

    def test_fixture_seed_protocol_and_stop_decay(self):
        a=list(ideal_sequence(SPEC,'noise_stop',80,43));b=list(ideal_sequence(SPEC,'noise_stop',80,43))
        for (x,tx),(y,ty) in zip(a,b):
            np.testing.assert_array_equal(x,y);self.assertEqual(tx,ty)
        self.assertTrue(np.count_nonzero(a[60][0])==0)
        cfg=ValidityConfig(mode='geometry_support')
        model=TensorMSTd(ArrayOps(),SPEC,horizontal_centers(SPEC,np.linspace(-20,20,9)),validity_config=cfg)
        for _ in range(100):model.step(dict(vectors=a[0][0],evidence=.7),.005)
        before=float(model.activity.max())
        for _ in range(400):state=model.step(dict(vectors=np.zeros_like(a[0][0]),evidence=0),.005)
        self.assertFalse(state['valid']);self.assertLess(float(model.activity.max()),before*.01)

    def test_latency_uses_first_sustained_valid_window_and_keeps_invalid_errors(self):
        rows=[dict(frame=i,heading_deg=10.,candidate_heading_deg=0.,bearing_valid=False,alpha_x_deg=None,
                   phase='clean',activity=[.1,.2]) for i in range(50)]
        for i in range(43,50):rows[i].update(bearing_valid=True,alpha_x_deg=10.,candidate_heading_deg=10.)
        result=response_metrics(rows,case='step')
        self.assertAlmostEqual(result['switch_settle_s'],.2)
        self.assertGreater(result['candidate_mae_deg'],0)
        self.assertEqual(result['accepted_mae_deg'],0)

    @unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch unavailable')
    def test_torch_dynamics_match_numpy(self):
        import torch
        for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
            op=ArrayOps('torch',device);v,_=next(ideal_sequence(SPEC,'hold',80))
            for cfg in temporal_variants().values():
                a=TensorMSTd(ArrayOps(),SPEC,horizontal_centers(SPEC,np.linspace(-20,20,9)),cfg)
                b=TensorMSTd(op,SPEC,horizontal_centers(SPEC,np.linspace(-20,20,9)),cfg)
                for _ in range(20):
                    a.step(dict(vectors=v,evidence=.7),.005)
                    b.step(dict(vectors=op.array(v),evidence=.7),.005)
                np.testing.assert_allclose(a.activity,op.host(b.activity),atol=2e-6)


if __name__=='__main__':unittest.main()
