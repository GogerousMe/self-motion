"""Acceptance regressions independent of V1/MT tuning and neural dynamics."""
from dataclasses import asdict
import importlib.util
from pathlib import Path
import unittest

import numpy as np

from mstd.data import V1Spec
from mstd.model import VisionModel, model_config
from mstd.mt import MTField
from mstd.mstd import DynamicsConfig, MSTd, horizontal_centers
from mstd.tensor_pipeline import TensorMSTd
from mstd.v1 import ArrayOps
from mstd.validity import ValidityConfig


class ValidityTests(unittest.TestCase):
    def setUp(self):
        self.spec = V1Spec(40, 40, tuple(np.arange(8)*np.pi/4), 20., 20., 19.5, 19.5)
        self.centers = horizontal_centers(self.spec, range(-20, 21, 5))
        self.gate = ValidityConfig(mode='geometry_support')

    def radial(self, angle=7, strength=.5):
        y, x = np.indices((40, 40))
        theta = np.arctan2(y-19.5, x-(19.5+20*np.tan(np.radians(angle))))
        return (strength*np.stack((np.cos(theta), np.sin(theta)))).astype(np.float32)

    def state(self, v):
        return dict(vectors=v, evidence=float(np.hypot(v[0], v[1]).mean()))

    def test_off_grid_acceptance_does_not_change_dynamics_or_legacy_decision(self):
        old = TensorMSTd(ArrayOps(), self.spec, self.centers)
        new = TensorMSTd(ArrayOps(), self.spec, self.centers, validity_config=self.gate)
        mt = self.state(self.radial())
        for _ in range(80):
            a, b = old.step(mt, .05), new.step(mt, .05)
            self.assertEqual(a['activity'], b['activity'])
            self.assertEqual(a['template_scores'], b['template_scores'])
            self.assertEqual(a['valid'], b['validity']['legacy_valid'])
        self.assertFalse(a['valid'])
        self.assertTrue(b['bearing_valid'])
        self.assertLessEqual(abs(b['alpha_x_deg']-7), 2.5)

    def test_no_structure_weak_input_and_concentrated_patch_are_rejected(self):
        rng = np.random.default_rng(818)
        theta = rng.uniform(-np.pi, np.pi, (40, 40))
        random = .5*np.stack((np.cos(theta), np.sin(theta)))
        uniform = np.zeros((2, 40, 40), np.float32); uniform[0] = .5
        sparse = self.radial(0); sparse[:, 8:, :] = 0
        weak = self.radial(0, .005)
        for v in (random, uniform, sparse, weak, self.radial(0, 0)):
            with self.subTest(evidence=self.state(v)['evidence']):
                model = TensorMSTd(ArrayOps(), self.spec, self.centers, validity_config=self.gate)
                # Charge activity first: absence of current support must reject stale activity.
                model.step(self.state(self.radial(0)), .5)
                self.assertFalse(model.step(self.state(v), .05)['valid'])

    def test_list_numpy_and_optional_torch_share_geometry_semantics(self):
        backends = ['numpy'] + (['torch'] if importlib.util.find_spec('torch') else [])
        ref = MSTd(self.spec, self.centers, validity_config=self.gate)
        models = [TensorMSTd(ArrayOps(b), self.spec, self.centers, validity_config=self.gate)
                  for b in backends]
        rng = np.random.default_rng(19)
        for v in (self.radial(0), self.radial(-12),
                  rng.normal(0, .1, (2, 40, 40)).astype(np.float32), self.radial(0, 0)):
            field = self.state(v)
            result = ref.step(MTField(tuple(v[0].flat), tuple(v[1].flat), field['evidence']), .05)
            for model in models:
                state = model.step({**field, 'vectors':model.ops.array(v)}, .05)
                np.testing.assert_allclose(state['activity'], result['activity'], atol=3e-6)
                self.assertEqual(state['valid'], result['valid'])
                for key in ('uniform_score', 'geometry_excess', 'active_fraction'):
                    self.assertAlmostEqual(state['validity'][key], result['validity'][key], places=5)

    def test_config_is_opt_in_validated_and_reaches_the_model(self):
        self.assertEqual(model_config()['mstd_validity']['mode'], 'legacy_margin')
        for settings in ({'mode':'unknown'}, {'min_template_score':float('nan')},
                         {'min_active_fraction':1.1}, {'min_geometry_excess':True},
                         {'local_evidence_threshold':0}, {'typo':.2}):
            with self.assertRaises(ValueError):
                model_config({'mstd_validity':settings})
        config = model_config({'projection_config':str(Path(__file__).resolve().parents[1]/'configs/v1_projection.json'),
                               'mstd_validity':asdict(self.gate)})
        model = VisionModel(19, 21, config)
        self.assertEqual(model.mstd.validity_config, self.gate)
        # Uncalibrated geometric support must not publish an angle.
        y, x = np.indices((21, 19)); theta = np.arctan2(y-10, x-9)
        v = .5*np.stack((np.cos(theta), np.sin(theta))).astype(np.float32)
        state = model.mstd.step(self.state(v), .5)
        self.assertTrue(state['valid'])
        self.assertFalse(state['bearing_valid'])
        self.assertIsNone(state['alpha_x_deg'])

    def test_spiking_image_chain_keeps_identical_neural_outputs(self):
        config = dict(mt_backend='python_spiking', mt_radius=0, backend='numpy', device='cpu',
                      projection_config=str(Path(__file__).resolve().parents[1]/'configs/v1_projection.json'),
                      intrinsics=dict(fx_px=10., fy_px=10., cx_px=9., cy_px=10.))
        old = VisionModel(19, 21, config)
        new = VisionModel(19, 21, {**config, 'mstd_validity':asdict(self.gate)})
        rng = np.random.default_rng(672)
        for i in range(3):
            image = rng.integers(0, 256, (21, 19), dtype=np.uint8)
            a, b = old.step(image, i*.05), new.step(image, i*.05)
            for key in ('exc_counts', 'inh_counts', 'activity', 'vectors'):
                np.testing.assert_array_equal(a['mt'][key], b['mt'][key])
            self.assertEqual(a['mstd']['activity'], b['mstd']['activity'])
            self.assertEqual(a['mstd']['valid'], b['mstd']['validity']['legacy_valid'])


if __name__ == '__main__':
    unittest.main()
