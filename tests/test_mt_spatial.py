"""Regression checks for spatial geometry, delay causality, and normal entry."""
import importlib.util
import json
from math import exp
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from mstd.calibration import stimulus_frames, write_stimulus
from mstd.model import VisionModel, model_config
from mstd.mt_circuit import DirectionSpatialKernel, SpatialConfig
from mstd.neural import SpikeConfig
from mstd.projection import load_projection
from mstd.simulation import simulate
from mstd.spiking_mt import PythonSpikingMT
from mstd.v1 import ArrayOps


ROOT = Path(__file__).resolve().parents[1]
PROJECTION = str(ROOT / 'configs/v1_projection.json')
ANGLES = np.arange(8) * np.pi / 4


class SpatialKernelTests(unittest.TestCase):
    def kernel(self, width, height, **settings):
        return DirectionSpatialKernel(ArrayOps(), ANGLES, width, height, SpatialConfig(**settings))

    def test_constants_preserved_at_edges_corners_and_tiny_grids(self):
        for width, height in ((1, 1), (2, 5), (5, 2), (9, 7)):
            kernel = self.kernel(width, height, radius=4)
            values = np.broadcast_to(np.arange(1, 9)[:, None, None], kernel.shape).astype(np.float32)
            np.testing.assert_allclose(kernel.apply(values), values, atol=2e-6, rtol=2e-6)

    def test_rotation_and_reflection_equivariance(self):
        values = np.random.default_rng(1).random((8, 7, 9), dtype=np.float32)
        expected = self.kernel(9, 7).apply(values)
        for turns in (1, 2, 3):
            rotated = np.roll(np.rot90(values, turns, axes=(1, 2)), -2 * turns, axis=0).copy()
            kernel = self.kernel(rotated.shape[2], rotated.shape[1])
            transformed = np.roll(np.rot90(expected, turns, axes=(1, 2)), -2 * turns, axis=0)
            np.testing.assert_allclose(kernel.apply(rotated), transformed, atol=2e-6, rtol=2e-6)
        reflected = values[[4, 3, 2, 1, 0, 7, 6, 5], :, ::-1].copy()
        np.testing.assert_allclose(self.kernel(9, 7).apply(reflected),
                                   expected[[4, 3, 2, 1, 0, 7, 6, 5], :, ::-1], atol=2e-6)

    def test_eight_kernels_have_correct_axis_and_opponent_geometry(self):
        weights = self.kernel(9, 9).weights
        np.testing.assert_allclose(weights.sum(axis=(1, 2)), 1, atol=1e-6)
        np.testing.assert_allclose(weights[:4], weights[4:], atol=1e-7)
        self.assertGreater(weights[0, 2, 3], weights[0, 3, 2])
        self.assertGreater(weights[1, 3, 3], weights[1, 1, 3])
        self.assertGreater(weights[2, 3, 2], weights[2, 2, 3])
        self.assertGreater(weights[3, 1, 3], weights[3, 3, 3])
        circular = self.kernel(9, 9, sigma_parallel_px=1, sigma_perpendicular_px=1).weights
        np.testing.assert_allclose(circular, np.broadcast_to(circular[0], circular.shape), atol=1e-7)

    def test_impulse_stays_local_and_in_its_direction_channel(self):
        kernel = self.kernel(9, 7)
        values = np.zeros(kernel.shape, np.float32)
        values[1, 0, 0] = 1
        result = kernel.apply(values)
        self.assertEqual(np.count_nonzero(result[np.arange(8) != 1]), 0)
        self.assertEqual(np.count_nonzero(result[:, 3:, :]), 0)
        self.assertEqual(np.count_nonzero(result[:, :, 3:]), 0)
        self.assertGreater(result[1, 1, 1], 0)
        for bad in (values - 1, values * np.nan, values[:, :2]):
            with self.assertRaises(ValueError):
                kernel.apply(bad)

    def test_invalid_config_rejected_and_wrong_backend_rejected(self):
        for setting in ({'enabled': 1}, {'radius': 0}, {'delay_ticks': 0}, {'radius': True},
                        {'gain': -1}, {'gain': float('nan')}, {'tau_ms': 0},
                        {'sigma_parallel_px': float('inf')}, {'sigma_perpendicular_px': False}):
            with self.assertRaises(ValueError):
                SpatialConfig(**setting)
        with self.assertRaises(ValueError):
            model_config({'mt_spatial': {'enabled': True}})
        with self.assertRaises(ValueError):
            model_config({'mt_spatial': {'unknown': 1}})


class SpatialCircuitTests(unittest.TestCase):
    def circuit(self, spatial_config=None, **kwargs):
        return PythonSpikingMT(ArrayOps(), load_projection(PROJECTION), 3, 3,
                               SpikeConfig(substeps=32), spatial_config=spatial_config, **kwargs)

    def test_exact_delay_decay_neural_drive_and_reset(self):
        config = SpatialConfig(enabled=True, delay_ticks=2, gain=0.1)
        mt = self.circuit(config)
        zero = mt.op.zeros(mt.input_shape)
        mt.set_rates(zero)
        impulse = mt.op.zeros(mt.shape)
        impulse[0, 1, 1] = 1
        expected = mt.spatial.kernel.apply(impulse) * config.gain
        with patch.object(mt.exc, 'step', return_value=impulse):
            mt.tick(v1_events=zero)
        self.assertEqual(float(mt.spatial.synapse.g.sum()), 0)
        with patch.object(mt.exc, 'step', return_value=mt.zero):
            mt.tick(v1_events=zero)
        self.assertEqual(float(mt.spatial.synapse.g.sum()), 0)
        with patch.object(mt.exc, 'step', wraps=mt.exc.step) as step:
            mt.tick(v1_events=zero)
            np.testing.assert_allclose(step.call_args.args[0], expected, atol=1e-7)
        np.testing.assert_allclose(mt.spatial.synapse.g, expected, atol=1e-7)
        mt.tick(v1_events=zero)
        np.testing.assert_allclose(mt.spatial.synapse.g, expected * exp(-1 / config.tau_ms), atol=1e-7)
        diagnostics = mt.diagnostics()
        np.testing.assert_array_equal(diagnostics['exc_conductance'], diagnostics['spatial_exc_conductance'])
        diagnostics['spatial_exc_conductance'][:] = 999
        self.assertLess(float(mt.spatial.synapse.g.max()), 1)
        mt.reset()
        self.assertEqual(float(mt.spatial.synapse.g.sum()), 0)
        self.assertEqual(float(mt.spatial.delay.read().sum()), 0)

    def test_disabled_zero_gain_and_population_order_reproduce_reference(self):
        configs = (None, SpatialConfig(enabled=False, gain=20), SpatialConfig(enabled=True, gain=0))
        models = [self.circuit(config) for config in configs]
        raw = np.full(models[0].input_shape, 10, np.float32)
        for mt in models:
            mt.set_rates(raw)
        for _ in range(30):
            reference = models[0].tick()
            for mt in models[1:]:
                actual = mt.tick()
                for key in reference:
                    np.testing.assert_array_equal(reference[key], actual[key])
                np.testing.assert_array_equal(models[0].exc.v, mt.exc.v)
        a = self.circuit(SpatialConfig(enabled=True))
        b = self.circuit(SpatialConfig(enabled=True), population_order=('inh', 'exc'))
        for mt in (a, b):
            mt.set_rates(raw)
        for _ in range(30):
            sa, sb = a.tick(), b.tick()
            np.testing.assert_array_equal(sa['exc'], sb['exc'])
        np.testing.assert_array_equal(a.exc.v, b.exc.v)

    def test_normal_image_runner_monitors_replay_and_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stimulus = root / 'stim.dat'
            write_stimulus(stimulus, 19, 19, stimulus_frames(19, 19, 3, 0, 1, kind='texture'))
            settings = dict(mt_backend='python_spiking', mt_radius=0, backend='numpy', device='cpu',
                            projection_config=PROJECTION, frame_ms=5, spiking=dict(substeps=32),
                            mt_spatial=dict(enabled=True), progress_every=0,
                            input=dict(path=str(stimulus)), monitors=dict(neural_state=True))
            report = simulate({**settings, 'output': str(root / 'run')})
            self.assertTrue(report['complete'])
            self.assertTrue(report['model']['mt_spatial']['enabled'])
            monitor = root / 'run/monitors/mt_spatial_exc_conductance'
            meta = json.loads((monitor / 'metadata.json').read_text())
            self.assertEqual(meta['shape'], [3, 8, 19, 19])
            self.assertEqual(meta['units'], 'effective conductance')
            values = np.fromfile(monitor / 'values.f32', dtype='<f4')
            self.assertTrue(np.isfinite(values).all())
            replay = simulate({**settings, 'output': str(root / 'replay'),
                               'input': dict(kind='v1_recording', path=str(root / 'run/monitors/v1')),
                               'layers': dict(v1=False)})
            self.assertEqual(report['last_mstd'], replay['last_mstd'])
            for name in ('mt_exc_rates', 'mt_readout_activity', 'mt_spatial_exc_conductance'):
                self.assertEqual((root / f'run/monitors/{name}/values.f32').read_bytes(),
                                 (root / f'replay/monitors/{name}/values.f32').read_bytes())
            model = VisionModel(3, 3, {**settings, 'input': dict(kind='v1_recording'), 'layers': dict(v1=False)})
            raw = np.ones((3, 28, 3, 3), np.float32)
            state = model.step(raw, 0)
            model.reset()
            repeated = model.step(raw, 0)
            np.testing.assert_array_equal(state['mt']['exc_counts'], repeated['mt']['exc_counts'])

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch not installed')
    def test_torch_kernels_and_fixed_event_circuit_match_numpy(self):
        import torch
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            op = ArrayOps('torch', device)
            config = SpatialConfig(enabled=True)
            kernel = DirectionSpatialKernel(op, ANGLES, 7, 5, config)
            values = np.random.default_rng(8).random((8, 5, 7), dtype=np.float32)
            reference = DirectionSpatialKernel(ArrayOps(), ANGLES, 7, 5, config).apply(values)
            np.testing.assert_allclose(op.host(kernel.apply(op.array(values))), reference, atol=2e-6)
            a = self.circuit(config)
            b = PythonSpikingMT(op, load_projection(PROJECTION), 3, 3,
                                SpikeConfig(substeps=32), spatial_config=config)
            raw = np.zeros(a.input_shape, np.float32)
            a.set_rates(raw)
            b.set_rates(op.array(raw))
            for tick in range(50):
                events = raw.copy()
                if tick % 5 == 0:
                    events[0, 0, 1, 1] = 1
                sa, sb = a.tick(v1_events=events), b.tick(v1_events=op.array(events))
                np.testing.assert_array_equal(sa['exc'], op.host(sb['exc']))
                np.testing.assert_allclose(a.exc.v, op.host(b.exc.v), atol=1e-3, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
