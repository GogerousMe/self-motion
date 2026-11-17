"""Direction geometry, causal J neurons, and normal simulation regression."""
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
from mstd.mt_circuit import CompetitionConfig, DirectionCompetition, SpatialConfig, direction_inhibition_matrix
from mstd.neural import SpikeConfig
from mstd.projection import Projection
from mstd.simulation import simulate
from mstd.spiking_mt import PythonSpikingMT
from mstd.v1 import ArrayOps


ANGLES = np.arange(8) * np.pi / 4
ROOT = Path(__file__).resolve().parents[1]
PROJECTION = str(ROOT / 'configs/v1_projection.json')


def projection():
    weights = np.zeros((28, 8))
    weights[0, 0] = 1
    weights[1, 4] = 1
    return Projection('competition_test', tuple(map(tuple, weights)), (1., 1., 1.), tuple(ANGLES), 1.)


class CompetitionTests(unittest.TestCase):
    def circuit(self, config=None, **kwargs):
        return PythonSpikingMT(ArrayOps(), projection(), 2, 2, SpikeConfig(seed=43, substeps=32),
                               competition_config=config, **kwargs)

    def test_matrix_normalization_rotation_mirror_and_opponent(self):
        weights = direction_inhibition_matrix(ANGLES)
        np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-7)
        np.testing.assert_array_equal(weights.diagonal(), 0)
        self.assertTrue((weights >= 0).all())
        self.assertTrue(np.all(np.diff(weights[0, :5]) > 0))
        for shift in range(8):
            np.testing.assert_allclose(np.roll(weights, (shift, shift), axis=(0, 1)), weights, atol=1e-7)
        mirror = [4, 3, 2, 1, 0, 7, 6, 5]
        np.testing.assert_allclose(weights[np.ix_(mirror, mirror)], weights, atol=1e-7)
        np.testing.assert_allclose(direction_inhibition_matrix(ANGLES + .37), weights, atol=1e-7)
        expected = np.zeros((8, 8), np.float32)
        expected[np.arange(8), (np.arange(8) + 4) % 8] = 1
        np.testing.assert_array_equal(direction_inhibition_matrix(ANGLES, 'opponent'), expected)

    def test_invalid_config_and_backend_rejected(self):
        for setting in ({'enabled': 1}, {'mode': 'unknown'}, {'input_weight': -1},
                        {'inhibition_weight': float('nan')}, {'tau_exc_ms': 0},
                        {'tau_inh_ms': True}, {'input_delay_ticks': 0}, {'output_delay_ticks': 1.5}):
            with self.assertRaises(ValueError):
                CompetitionConfig(**setting)
        for angles in (np.zeros(8), np.arange(7), ANGLES * np.nan):
            with self.assertRaises(ValueError):
                direction_inhibition_matrix(angles)
        with self.assertRaises(ValueError):
            direction_inhibition_matrix(ANGLES * .5, 'opponent')
        with self.assertRaises(ValueError):
            model_config({'mt_competition': {'enabled': True}})
        with self.assertRaises(ValueError):
            model_config({'mt_competition': {'unknown': True}})

    def test_two_delay_legs_decay_locality_and_direction_routing(self):
        op = ArrayOps()
        config = CompetitionConfig(enabled=True, input_delay_ticks=2, output_delay_ticks=3)
        circuit = DirectionCompetition(op, ANGLES, 3, 3, config)
        zero = op.zeros(circuit.shape)
        impulse = zero.copy()
        impulse[0, 1, 1] = 1
        ge, gi = circuit.arrive(1)
        self.assertEqual(float(ge.sum() + gi.sum()), 0)
        circuit.emit(impulse, impulse)
        expected_ge = impulse * config.input_weight
        expected_gi = np.zeros(circuit.shape, np.float32)
        expected_gi[:, 1, 1] = circuit.weights[:, 0] * config.inhibition_weight
        for tick in range(1, 5):
            ge, gi = circuit.arrive(1)
            np.testing.assert_allclose(ge, expected_ge * exp(-(tick - 2) / config.tau_exc_ms) if tick >= 2 else zero, atol=1e-7)
            np.testing.assert_allclose(gi, expected_gi * exp(-(tick - 3) / config.tau_inh_ms) if tick >= 3 else zero, atol=1e-7)
            circuit.emit(zero, zero)
        self.assertEqual(gi[0, 1, 1], 0)
        self.assertGreater(gi[4, 1, 1], gi[1, 1, 1])
        self.assertEqual(np.count_nonzero(gi[:, 0, :]), 0)
        circuit.reset()
        self.assertEqual(float(circuit.exc_synapse.g.sum() + circuit.inh_synapse.g.sum()), 0)

    def test_e_to_j_to_e_timing_in_normal_tick(self):
        mt = self.circuit(CompetitionConfig(enabled=True))
        raw = mt.op.zeros(mt.input_shape)
        mt.set_rates(raw)
        impulse = mt.op.zeros(mt.shape)
        impulse[0, 0, 0] = 1
        seen = []
        def j_step(ge, gi, **kwargs):
            seen.append(ge.copy())
            return impulse if ge.sum() else mt.zero
        with patch.object(mt.competition.population, 'step', side_effect=j_step):
            with patch.object(mt.exc, 'step', return_value=impulse):
                mt.tick(v1_events=raw)
            self.assertEqual(float(mt.competition.inh_synapse.g.sum()), 0)
            with patch.object(mt.exc, 'step', return_value=mt.zero):
                mt.tick(v1_events=raw)
            self.assertEqual(float(mt.competition.inh_synapse.g.sum()), 0)
            with patch.object(mt.exc, 'step', wraps=mt.exc.step) as exc_step:
                mt.tick(v1_events=raw)
                self.assertGreater(exc_step.call_args.args[1][4, 0, 0], 0)
        self.assertEqual(float(seen[0].sum()), 0)
        self.assertGreater(float(seen[1].sum()), 0)

    def test_disabled_and_zero_output_gain_preserve_baseline(self):
        models = [self.circuit(), self.circuit(CompetitionConfig(enabled=False)),
                  self.circuit(CompetitionConfig(enabled=True, inhibition_weight=0))]
        raw = np.zeros(models[0].input_shape, np.float32)
        raw[0, :2] = 100
        for mt in models:
            mt.set_rates(raw)
        for _ in range(100):
            reference = models[0].tick()
            for mt in models[1:]:
                state = mt.tick()
                for key in reference:
                    np.testing.assert_array_equal(reference[key], state[key])
                np.testing.assert_array_equal(models[0].exc.v, mt.exc.v)
                np.testing.assert_array_equal(models[0].trace, mt.trace)

    def test_population_order_silence_and_reset(self):
        config = CompetitionConfig(enabled=True)
        a = self.circuit(config)
        b = self.circuit(config, population_order=('competition', 'inh', 'exc'))
        raw = np.full(a.input_shape, 10, np.float32)
        for mt in (a, b):
            mt.set_rates(raw)
        for _ in range(80):
            sa, sb = a.tick(), b.tick()
            for key in sa:
                np.testing.assert_array_equal(sa[key], sb[key])
        np.testing.assert_array_equal(a.competition.population.v, b.competition.population.v)
        voltage = a.exc.v.copy()
        a.reset()
        a.set_rates(raw)
        for _ in range(80):
            a.tick()
        np.testing.assert_array_equal(a.exc.v, voltage)
        a.reset()
        a.set_rates(np.zeros(a.input_shape))
        for _ in range(100):
            state = a.tick()
            self.assertEqual(float(state['exc'].sum() + state['competition'].sum()), 0)

    def test_actual_spiking_competition_suppresses_weaker_opponent(self):
        base = self.circuit()
        competing = self.circuit(CompetitionConfig(enabled=True, mode='opponent'))
        raw = np.zeros(base.input_shape, np.float32)
        raw[0, 0] = 200
        raw[0, 1] = 80
        for mt in (base, competing):
            mt.set_rates(raw)
        totals = [np.zeros(base.shape), np.zeros(base.shape)]
        j_total = 0
        for tick in range(400):
            for index, mt in enumerate((base, competing)):
                state = mt.tick()
                if tick >= 100:
                    totals[index] += state['exc']
                if index == 1:
                    j_total += float(state['competition'].sum())
        self.assertGreater(j_total, 0)
        self.assertGreater(totals[1][0].sum(), 0)
        self.assertLess(totals[1][4].sum(), totals[0][4].sum())

    def test_symmetric_opponent_input_preserves_both_supported_directions(self):
        mt = self.circuit(CompetitionConfig(enabled=True, mode='opponent'))
        zero = mt.op.zeros(mt.input_shape)
        mt.set_rates(zero)
        totals = mt.op.zeros(mt.shape)
        for tick in range(120):
            events = zero.copy()
            if tick % 8 == 0:
                events[0, :2] = 1
            state = mt.tick(v1_events=events)
            totals += state['exc']
            np.testing.assert_array_equal(state['exc'][0], state['exc'][4])
        self.assertGreater(float(totals[0].sum()), 0)
        self.assertGreater(float(totals[4].sum()), 0)

    def test_frame_partition_preserves_j_events_and_state(self):
        config = dict(mt_backend='python_spiking', mt_radius=0, backend='numpy', device='cpu',
                      projection_config=PROJECTION, spiking=dict(substeps=32),
                      input=dict(kind='v1_recording'), layers=dict(v1=False), mt_competition=dict(enabled=True))
        a = VisionModel(2, 2, {**config, 'frame_ms':20})
        b = VisionModel(2, 2, {**config, 'frame_ms':10})
        raw = np.full((3, 28, 2, 2), 20, np.float32)
        whole = a.step(raw, 0)
        first = b.step(raw, 0)
        second = b.step(raw, .01)
        for key in ('exc_counts', 'inh_counts', 'competition_counts', 'v1_counts'):
            np.testing.assert_array_equal(whole['mt'][key], first['mt'][key] + second['mt'][key])
        np.testing.assert_array_equal(a.mt.competition.population.v, b.mt.competition.population.v)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch not installed')
    def test_torch_fixed_events_match_numpy_with_both_circuits(self):
        import torch
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            config = CompetitionConfig(enabled=True)
            a = self.circuit(config, spatial_config=SpatialConfig(enabled=True))
            op = ArrayOps('torch', device)
            b = PythonSpikingMT(op, projection(), 2, 2, SpikeConfig(substeps=32),
                                spatial_config=SpatialConfig(enabled=True), competition_config=config)
            zero = a.op.zeros(a.input_shape)
            a.set_rates(zero)
            b.set_rates(op.array(zero))
            for tick in range(60):
                events = zero.copy()
                if tick % 10 == 0:
                    events[0, :2] = 1
                sa, sb = a.tick(v1_events=events), b.tick(v1_events=op.array(events))
                for key in ('exc', 'inh', 'competition'):
                    np.testing.assert_array_equal(sa[key], op.host(sb[key]))
                np.testing.assert_allclose(a.exc.v, op.host(b.exc.v), atol=1e-3, rtol=1e-5)
                np.testing.assert_allclose(a.competition.inh_synapse.g, op.host(b.competition.inh_synapse.g), atol=1e-6)

    def test_normal_image_runner_j_events_replay_and_monitor_passivity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stimulus = root / 'stim.dat'
            write_stimulus(stimulus, 19, 19, stimulus_frames(19, 19, 3, 0, 1, kind='texture'))
            settings = dict(mt_backend='python_spiking', mt_radius=0, backend='numpy', device='cpu',
                            projection_config=PROJECTION, frame_ms=10, spiking=dict(substeps=32),
                            mt_competition=dict(enabled=True), mt_spatial=dict(enabled=True), progress_every=0,
                            input=dict(path=str(stimulus)), monitors=dict(neural_state=True, spikes=True))
            report = simulate({**settings, 'output': str(root / 'run')})
            self.assertTrue(report['complete'])
            monitor = root / 'run/monitors/mt_competition_rates'
            meta = json.loads((monitor / 'metadata.json').read_text())
            self.assertEqual(meta['shape'], [3, 8, 19, 19])
            self.assertEqual(meta['units'], 'Hz')
            dtype = np.dtype([('tick', '<u8'), ('neuron', '<u4'), ('count', '<u4')])
            events = np.fromfile(root / 'run/monitors/spikes/competition.events', dtype=dtype)
            observed = np.fromfile(monitor / 'values.f32', dtype='<f4').reshape(3, 8, 19, 19)
            reconstructed = np.zeros_like(observed)
            for event in events:
                reconstructed[int(event['tick']) // 10].reshape(-1)[int(event['neuron'])] += int(event['count']) * 100
            np.testing.assert_array_equal(reconstructed, observed)
            event_meta = json.loads((root / 'run/monitors/spikes/metadata.json').read_text())
            self.assertEqual(event_meta['shapes']['competition'], [8, 19, 19])
            replay = simulate({**settings, 'output': str(root / 'replay'),
                               'input': dict(kind='v1_recording', path=str(root / 'run/monitors/v1')),
                               'layers': dict(v1=False), 'monitors': dict(neural_state=False, spikes=False)})
            self.assertEqual(report['last_mstd'], replay['last_mstd'])
            for name in ('mt_exc_rates', 'mt_competition_rates', 'mt_readout_activity'):
                self.assertEqual((root / f'run/monitors/{name}/values.f32').read_bytes(),
                                 (root / f'replay/monitors/{name}/values.f32').read_bytes())


if __name__ == '__main__':
    unittest.main()
