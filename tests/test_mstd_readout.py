"""Readout regression: angular geometry, safe fallbacks and unchanged dynamics."""
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import sys

import numpy as np

from mstd.data import V1Spec
from mstd.model import VisionModel,model_config
from mstd.mt import MTField
from mstd.mstd import MSTd,horizontal_centers
from mstd.readout import ReadoutConfig,heading_readout
from mstd.tensor_pipeline import TensorMSTd
from mstd.v1 import ArrayOps
from mstd.validity import ValidityConfig


class ReadoutTests(unittest.TestCase):
    def setUp(self):
        self.spec=V1Spec(40,40,tuple(np.arange(8)*np.pi/4),20.,20.,19.5,19.5)
        self.angles=np.arange(-20,21,5)
        self.centers=horizontal_centers(self.spec,self.angles)
        self.cfg=ReadoutConfig(mode='local_quadratic')

    def read(self,activity,winner=5,valid=True,centers=None,spec=None):
        return heading_readout(activity,self.centers if centers is None else centers,
                               self.spec if spec is None else spec,winner,valid,self.cfg)

    def test_quadratic_uses_angular_spacing_and_preserves_grid_angle(self):
        activity=(1-.001*(self.angles-7)**2).tolist()
        result=self.read(activity)
        self.assertAlmostEqual(result['alpha_x_deg'],7,places=12)
        self.assertAlmostEqual(result['winner_alpha_x_deg'],5,places=12)
        self.assertEqual(result['readout']['method_used'],'local_quadratic')
        # In this angular grid, pixel-center spacings are deliberately nonuniform.
        self.assertNotAlmostEqual(self.centers[1][0]-self.centers[0][0],self.centers[4][0]-self.centers[3][0])

    def test_invalid_and_uncalibrated_inputs_publish_no_bearing(self):
        activity=(1-.001*(self.angles-7)**2).tolist()
        for result in (self.read(activity,valid=False),self.read(activity,spec=SimpleNamespace(fx_px=None,fy_px=None,cx_px=19.5,cy_px=19.5))):
            self.assertFalse(result['bearing_valid'])
            for key in ('alpha_x_deg','alpha_y_deg','winner_alpha_x_deg','local_quadratic_alpha_x_deg'):
                self.assertIsNone(result[key])
            json.dumps(result,allow_nan=False)

    def test_boundary_flat_and_unsupported_grids_fall_back(self):
        result=self.read(list(range(9)),winner=8)
        self.assertEqual(result['readout']['fallback_reason'],'boundary_peak')
        self.assertAlmostEqual(result['alpha_x_deg'],20)
        result=self.read([1]*9,winner=4)
        self.assertEqual(result['readout']['fallback_reason'],'flat_or_nonconcave_peak')
        nonuniform=horizontal_centers(self.spec,[-20,-15,-10,-5,0,4,10,15,20])
        result=self.read([0,0,0,0,1,3,2,0,0],centers=nonuniform)
        self.assertEqual(result['readout']['fallback_reason'],'non_uniform_angular_grid')
        nonhorizontal=list(self.centers);nonhorizontal[0]=(nonhorizontal[0][0],18)
        self.assertEqual(self.read([1]*9,centers=nonhorizontal)['readout']['fallback_reason'],'non_horizontal_grid')

    def test_reference_and_tensor_agree_without_changing_state_or_gate(self):
        y,x=np.indices((40,40));theta=np.arctan2(y-19.5,x-(19.5+20*np.tan(np.radians(7))))
        v=(.5*np.stack((np.cos(theta),np.sin(theta)))).astype(np.float32)
        field=dict(vectors=v,evidence=float(np.hypot(*v).mean()))
        gate=ValidityConfig(mode='geometry_support')
        old=TensorMSTd(ArrayOps(),self.spec,self.centers,validity_config=gate)
        new=TensorMSTd(ArrayOps(),self.spec,self.centers,validity_config=gate,readout_config=self.cfg)
        ref=MSTd(self.spec,self.centers,validity_config=gate,readout_config=self.cfg)
        mt=MTField(tuple(v[0].flat),tuple(v[1].flat),field['evidence'])
        for _ in range(30):
            a=old.step(field,.05);b=new.step(field,.05);c=ref.step(mt,.05)
            self.assertEqual(a['activity'],b['activity'])
            self.assertEqual(a['template_scores'],b['template_scores'])
            self.assertEqual(a['validity'],b['validity'])
            np.testing.assert_allclose(b['activity'],c['activity'],atol=3e-6,rtol=0)
            self.assertAlmostEqual(b['alpha_x_deg'],c['alpha_x_deg'],delta=1e-3)
        self.assertLess(abs(b['alpha_x_deg']-7),abs(a['alpha_x_deg']-7))

    def test_config_defaults_and_actual_neural_chain_remain_identical(self):
        self.assertEqual(model_config()['mstd_readout']['mode'],'winner')
        for settings in ({'mode':'typo'},{'curvature_epsilon':0},{'curvature_epsilon':True},{'curvature_epsilon':float('nan')}):
            with self.assertRaises(ValueError):model_config({'mstd_readout':settings})
        config=dict(mt_backend='python_spiking',mt_radius=0,backend='numpy',device='cpu',
                    projection_config=str(Path(__file__).resolve().parents[1]/'configs/v1_projection.json'),
                    intrinsics=dict(fx_px=10.,fy_px=10.,cx_px=9.,cy_px=10.))
        old=VisionModel(19,21,config)
        new=VisionModel(19,21,{**config,'mstd_readout':asdict(self.cfg)})
        for i in range(3):
            image=np.random.default_rng(i).integers(0,256,(21,19),dtype=np.uint8)
            a=old.step(image,i*.05);b=new.step(image,i*.05)
            np.testing.assert_array_equal(a['mt']['exc_counts'],b['mt']['exc_counts'])
            np.testing.assert_array_equal(a['mt']['inh_counts'],b['mt']['inh_counts'])
            self.assertEqual(a['mstd']['activity'],b['mstd']['activity'])
            self.assertEqual(a['mstd']['valid'],b['mstd']['valid'])

    def test_validation_reports_the_selected_readout_without_hiding_winner_error(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples'))
        from check_mstd_validity import make_row,summarize
        state=dict(activity=(1-.001*(self.angles-7.6)**2).tolist(),winner_index=6,
            template_scores=[.4]*9,valid=True,margin=.01,mt_evidence=.5,
            readout=asdict(self.cfg),validity=dict(legacy_valid=True,winner_template_score=.4,max_template_score=.4,
                uniform_score=.01,geometry_excess=.39,active_fraction=1.,rejected_by=[]))
        rows=[make_row('readout',i,state) for i in range(80)]
        result=summarize(dict(name='readout',expected_valid=True,heading=7),rows)
        self.assertAlmostEqual(result['unconditional_winner_mae_deg'],3)
        self.assertAlmostEqual(result['unconditional_selected_mae_deg'],.6)
        self.assertTrue(result['diagnostic_passed'])


if __name__=='__main__':unittest.main()
