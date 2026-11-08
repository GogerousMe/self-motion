"""Execution policy, deferred checks and device-resident neural integration."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from mstd.cli import main
from mstd.execution import execution_settings
from mstd.model import model_config,VisionModel
from mstd.neural import finite,IzhikevichPopulation
from mstd.spiking_mt import PythonSpikingMT
from mstd.v1 import ArrayOps
from mstd.calibration import write_stimulus,stimulus_frames
from test_python_spiking import projection

ROOT=Path(__file__).resolve().parents[1]


class ExecutionPolicyTests(unittest.TestCase):
    def test_default_gpu_and_explicit_cpu(self):
        self.assertEqual(execution_settings(),dict(backend='torch',device='cuda'))
        for options in (dict(device='cpu'),dict(backend='numpy')):
            self.assertEqual(execution_settings(**options),dict(backend='numpy',device='cpu'))
        self.assertEqual(execution_settings('torch','cpu'),dict(backend='torch',device='cpu'))
        self.assertEqual(execution_settings(device='gpu')['device'],'cuda')
        self.assertEqual(execution_settings(device='cuda:1')['device'],'cuda:1')
        for pair in (('numpy','cuda'),('torch','mps'),('torch','cuda:bad')):
            with self.assertRaises(ValueError):execution_settings(*pair)

    def test_spiking_defaults_and_legacy_rate_config(self):
        cfg=model_config(dict(mt_backend='python_spiking',mt_radius=0))
        self.assertEqual((cfg['backend'],cfg['device']),('torch','cuda'))
        cfg=model_config(dict(mt_backend='python_spiking',mt_radius=0,device='cpu'))
        self.assertEqual((cfg['backend'],cfg['device']),('numpy','cpu'))
        self.assertEqual(model_config()['device'],'cpu')

    def test_cli_single_cpu_option_overrides_gpu_template(self):
        report=dict(frames=2,model=dict(layers={},bearing_calibrated=False))
        for flags,expected in (([],('torch','cuda')),(['--device','cpu'],('numpy','cpu')),
                               (['--backend','torch','--device','cpu'],('torch','cpu'))):
            with patch('mstd.simulation.simulate',return_value=report) as run,redirect_stdout(io.StringIO()):
                main(['simulate','--config',str(ROOT/'configs/model_python_spiking.json'),*flags])
            config=run.call_args.args[0]
            self.assertEqual((config['backend'],config['device']),expected)
            self.assertEqual(config['mt_backend'],'python_spiking')

    def test_deferred_check_retains_failure_and_cleans_up(self):
        op=ArrayOps()
        value=np.array([np.inf],np.float32)
        with self.assertRaises(FloatingPointError),op.validation_scope():
            self.assertTrue(finite(op,value))
            value[:]=0  # Later reset must not erase an earlier overflow flag.
            self.assertTrue(finite(op,value))
        self.assertFalse(finite(op,np.array([np.nan])))
        with self.assertRaisesRegex(ValueError,'stop'),op.validation_scope():raise ValueError('stop')
        with op.validation_scope():self.assertTrue(finite(op,value))

    def test_deferred_mstd_export_keeps_all_neural_updates(self):
        config=dict(mt_backend='python_spiking',mt_radius=0,device='cpu',frame_ms=1,
                    input=dict(kind='v1_recording'),layers=dict(v1=False),
                    projection_config=str(ROOT/'configs/v1_projection.json'))
        a=VisionModel(2,2,config);b=VisionModel(2,2,config)
        signal=dict(vectors=np.ones((2,2,2),np.float32),evidence=1.)
        for _ in range(10):
            a.mstd.step(signal,.001)
            self.assertIsNone(b.mstd.step(signal,.001,return_state=False))
        np.testing.assert_array_equal(a.mstd.activity,b.mstd.activity)
        self.assertEqual(a.mstd.step(signal,.001),b.mstd.step(signal,.001))

    def test_cli_cpu_executes_gpu_template_with_native_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);stimulus=root/'image.dat';output=root/'run'
            write_stimulus(stimulus,19,19,stimulus_frames(19,19,2,0,1.5))
            with redirect_stdout(io.StringIO()):
                main(['simulate','--config',str(ROOT/'configs/model_python_spiking.json'),
                      '--device','cpu','--stimulus',str(stimulus),'--output',str(output),'--max-frames','2'])
            report=json.loads((output/'model_report.json').read_text())
            self.assertTrue(report['complete']);self.assertEqual(report['frames'],2)
            self.assertEqual((report['model']['backend'],report['model']['device']),('numpy','cpu'))
            self.assertTrue((output/'monitors/mt_exc_rates/values.f32').is_file())


@unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch not installed')
class CUDAExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        if not torch.cuda.is_available():raise unittest.SkipTest('CUDA not available')

    def test_fixed_events_match_cpu_neurons_and_synapses(self):
        cpu=PythonSpikingMT(ArrayOps(),projection(),2,2)
        gpu=PythonSpikingMT(ArrayOps('torch','cuda'),projection(),2,2)
        raw=np.zeros(cpu.input_shape,np.float32)
        cpu.set_rates(raw);gpu.set_rates(gpu.op.array(raw))
        # Exercise both signed projections and delayed I->E events without RNG differences.
        with gpu.op.validation_scope():
            for t in range(60):
                events=raw.copy()
                if t%10==0:events[0,:2,0,0]=1
                a=cpu.tick(v1_events=events);b=gpu.tick(v1_events=gpu.op.array(events))
                for key in ('exc','inh'):np.testing.assert_array_equal(a[key],gpu.op.host(b[key]))
                for population in ('exc','inh'):
                    np.testing.assert_allclose(getattr(cpu,population).v,gpu.op.host(getattr(gpu,population).v),atol=1e-3,rtol=1e-5)
            np.testing.assert_allclose(cpu.gi.g,gpu.op.host(gpu.gi.g),atol=1e-5,rtol=1e-5)

    def test_real_cuda_frame_readout_observer_and_feedback(self):
        cfg=dict(mt_backend='python_spiking',mt_radius=0,frame_ms=5,backend='torch',device='cuda',
                 input=dict(kind='v1_recording'),layers=dict(v1=False),
                 projection_config=str(ROOT/'configs/v1_projection.json'))
        a=VisionModel(2,2,cfg);b=VisionModel(2,2,cfg)
        seen=[]
        b.set_spike_observer(lambda tick,stamp,events:seen.append(tick))
        # A harmless controller forces per-tick host exports; results must match deferred exports.
        b.set_feedback(lambda ctx:None)
        raw=np.full((3,28,2,2),20.,np.float32)
        for t in range(3):
            sa=a.step(raw,t*.005);sb=b.step(raw,t*.005)
            self.assertIsInstance(sa['mt']['evidence'],float)
            self.assertEqual(sa['mstd'],sb['mstd'])
            np.testing.assert_array_equal(a.ops.host(sa['mt']['exc_counts']),b.ops.host(sb['mt']['exc_counts']))
        self.assertEqual(seen,list(range(15)))
        self.assertTrue(a.mt.exc.v.is_cuda);self.assertTrue(a.mstd.activity.is_cuda)
        with self.assertRaises(ValueError):a.step(np.full_like(raw,-1),.015)

    def test_cuda_overflow_is_detected_before_reset(self):
        op=ArrayOps('torch','cuda');neuron=IzhikevichPopulation(op,(1,))
        with self.assertRaises(FloatingPointError),op.validation_scope():
            neuron.step(op.zeros((1,))+1e38,op.zeros((1,)),1)

    def test_images_advance_entire_cuda_pipeline(self):
        cfg=dict(mt_backend='python_spiking',mt_radius=0,device='cuda',frame_ms=5,
                 projection_config=str(ROOT/'configs/v1_projection.json'))
        model=VisionModel(19,19,cfg)
        for i,image in enumerate(stimulus_frames(19,19,4,0,1.5)):
            state=model.step(np.frombuffer(image,dtype=np.uint8).reshape(19,19),i*.005)
            for value in (state['v1'],state['mt']['direction_rates'],model.mt.exc.v,model.mstd.activity):
                self.assertTrue(value.is_cuda)
                self.assertTrue(bool(model.ops.torch.isfinite(value).all()))
            self.assertIsInstance(state['mstd']['activity'],list)
        self.assertEqual(model.ticks,20)


if __name__=='__main__':unittest.main()
