"""Mechanism, causal-feedback, and runner tests for the Python neural backend."""
import importlib.util
import json
from math import exp
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from mstd.v1 import ArrayOps
from mstd.neural import SpikeConfig,DelayLine,PoissonEncoder,ExpConductance,IzhikevichPopulation
from mstd.spiking_mt import PythonSpikingMT
from mstd.projection import Projection
from mstd.model import VisionModel,model_config
from mstd.simulation import simulate
from mstd.calibration import stimulus_frames,write_stimulus

PROJECTION=str(Path(__file__).resolve().parents[1]/'configs/v1_projection.json')


def projection():
    weights=np.zeros((28,8));weights[0,0]=1;weights[1,0]=-1
    return Projection('test',tuple(map(tuple,weights)),(1.,1.,1.),tuple(np.arange(8)*np.pi/4),1.)


class NeuralTests(unittest.TestCase):
    def setUp(self):self.op=ArrayOps()

    def test_delay_exact_and_owned_buffers(self):
        line=DelayLine(self.op,(1,),2)
        self.assertEqual(line.read()[0],0);line.write(np.array([3.]))
        self.assertEqual(line.read()[0],0);line.write(np.array([4.]))
        observed=line.read();self.assertEqual(observed[0],3);observed[0]=99
        self.assertEqual(line.read()[0],3)
        line.reset();self.assertEqual(line.read()[0],0)

    def test_poisson_statistics_zero_high_rate_and_local_rng(self):
        encoder=PoissonEncoder(self.op,12)
        values=encoder.sample(np.full(50000,100.,np.float32),1000)
        self.assertLess(abs(values.mean()-100),.2)
        self.assertLess(abs(values.var()-100),3)
        encoder.reset();np.testing.assert_array_equal(values,encoder.sample(np.full(50000,100.,np.float32),1000))
        self.assertEqual(encoder.sample(np.zeros(100),1).sum(),0)
        high=encoder.sample(np.full(10000,10000.),1)
        self.assertLess(abs(high.mean()-10),.15);self.assertGreater(high.max(),1)
        for rates in (np.array([-1.]),np.array([np.inf])):
            with self.assertRaises(ValueError):encoder.sample(rates,1)

    def test_exponential_synapse_and_sign_of_reversal_current(self):
        syn=ExpConductance(self.op,(1,),5)
        self.assertEqual(syn.step(np.ones(1),1)[0],1)
        for _ in range(5):value=syn.step(np.zeros(1),1)
        self.assertAlmostEqual(value[0],exp(-1),places=6)
        zero=np.zeros(1);one=np.ones(1)*.1
        neurons=[IzhikevichPopulation(self.op,(1,)) for _ in range(3)]
        for neuron,ge,gi in zip(neurons,(zero,one,zero),(zero,zero,one)):neuron.step(ge,gi,1)
        self.assertGreater(neurons[1].v[0],neurons[0].v[0])
        self.assertLess(neurons[2].v[0],neurons[0].v[0])

    def test_neuron_spikes_reset_adaptation_and_substep_convergence(self):
        zero=np.zeros(1,np.float32)
        counts=[]
        for substeps in (4,8,16):
            neuron=IzhikevichPopulation(self.op,(1,))
            emitted=[]
            for t in range(1000):
                n=neuron.step(zero,zero,1,substeps,external=zero+10)
                if n[0]:emitted.append(t)
                self.assertLess(neuron.v[0],30)
            self.assertGreater(len(emitted),10)
            self.assertGreater(np.mean(np.diff(emitted)[-5:]),np.diff(emitted)[0])
            counts.append(len(emitted))
            neuron.reset();self.assertEqual(neuron.v[0],-65);self.assertEqual(neuron.u[0],-13)
            self.assertEqual(sum(neuron.step(zero,zero,1)[0] for _ in range(100)),0)
        self.assertLessEqual(max(counts)-min(counts),2)

    def test_mt_event_projection_delay_and_locality(self):
        mt=PythonSpikingMT(self.op,projection(),2,2)
        zero=self.op.zeros(mt.input_shape);mt.set_rates(zero)
        events=zero.copy();events[0,0,0,0]=2
        mt.tick(v1_events=events)
        self.assertEqual(mt.ge.g.sum(),0)
        mt.tick(v1_events=zero)
        self.assertEqual(mt.ge.g[0,0,0],3)
        self.assertEqual(np.count_nonzero(mt.ge.g),1)
        self.assertEqual(mt.drive_i.g.sum(),0)
        mt.reset();mt.set_rates(zero)
        events=zero.copy();events[0,1,0,0]=3
        previous_i=np.zeros(mt.shape);previous_g=np.zeros(mt.shape);total_i=0
        for t in range(60):
            sp=mt.tick(v1_events=events if t==0 else zero)
            np.testing.assert_allclose(mt.gi.g,previous_g*exp(-.1)+previous_i,atol=1e-6)
            total_i+=sp['inh'].sum();previous_i=sp['inh'].copy();previous_g=mt.gi.g.copy()
        self.assertGreater(total_i,0)
        self.assertEqual(mt.exc.v[:,1,:].max(),mt.exc.v[:,1,:].min())

    def test_config_rejects_invalid_steps_shapes_and_nonfinite_drive(self):
        for kw in (dict(dt_ms=0),dict(substeps=0),dict(delay_ticks=0),dict(seed=-1),dict(tau_exc_ms=float('nan'))):
            with self.assertRaises(ValueError):SpikeConfig(**kw)
        mt=PythonSpikingMT(self.op,projection(),2,2)
        with self.assertRaises(ValueError):mt.set_rates(np.zeros((84,2,2)))
        mt.set_rates(np.zeros(mt.input_shape))
        with self.assertRaises(ValueError):mt.tick(v1_events=np.full(mt.input_shape,.5))
        neuron=IzhikevichPopulation(self.op,(1,))
        with np.errstate(over='ignore'),self.assertRaises(FloatingPointError):
            neuron.step(np.full(1,1e38,np.float32),np.zeros(1,np.float32),1)

    def test_e_i_population_order_does_not_change_network(self):
        a=PythonSpikingMT(self.op,projection(),2,2)
        b=PythonSpikingMT(self.op,projection(),2,2,population_order=('inh','exc'))
        rates=np.zeros(a.input_shape,np.float32);rates[0,:2,0,0]=200
        a.set_rates(rates);b.set_rates(rates)
        for _ in range(100):
            sa=a.tick();sb=b.tick()
            for key in sa:np.testing.assert_array_equal(sa[key],sb[key])
        np.testing.assert_array_equal(a.exc.v,b.exc.v)
        np.testing.assert_array_equal(a.inh.u,b.inh.u)


class PythonNetworkTests(unittest.TestCase):
    def settings(self,**extra):
        return dict(mt_backend='python_spiking',mt_radius=0,projection_config=PROJECTION,
                    input=dict(kind='v1_recording',path='unused'),layers=dict(v1=False),progress_every=0,**extra)

    def test_feedback_is_delayed_and_reset_is_reproducible(self):
        cfg=self.settings(frame_ms=1,spiking=dict(feedback_delay_ticks=2))
        base=VisionModel(2,2,cfg);closed=VisionModel(2,2,cfg)
        contexts=[]
        def feedback(ctx):
            contexts.append(ctx)
            return dict(exc_current=np.full((8,2,2),10.))
        closed.set_feedback(feedback)
        raw=np.zeros((3,28,2,2),np.float32)
        for t in range(2):
            base.step(raw,t*.001);closed.step(raw,t*.001)
            np.testing.assert_array_equal(base.mt.exc.v,closed.mt.exc.v)
        base.step(raw,.002);closed.step(raw,.002)
        self.assertGreater(closed.mt.exc.v.mean(),base.mt.exc.v.mean())
        self.assertEqual(len(contexts),3)
        self.assertIsNotNone(contexts[0]['mstd'])
        self.assertEqual(contexts[0]['timestamp_s'],.001)
        state=closed.mt.exc.v.copy();closed.reset()
        for t in range(3):closed.step(raw,t*.001)
        np.testing.assert_array_equal(closed.mt.exc.v,state)

    def test_observers_and_controller_snapshots_cannot_mutate_neurons(self):
        a=VisionModel(2,2,self.settings(frame_ms=10))
        b=VisionModel(2,2,self.settings(frame_ms=10))
        raw=np.full((3,28,2,2),20.,np.float32)
        calls=[]
        def observe(t,stamp,spikes):
            calls.append(t)
            for v in spikes.values():v[:]=999
        def harmless(ctx):
            ctx['mt']['pooled_rates'][:]=999
            return None
        b.set_spike_observer(observe);b.set_feedback(harmless)
        for t in range(3):
            sa=a.step(raw,t*.01);sb=b.step(raw,t*.01)
            np.testing.assert_array_equal(sa['mt']['exc_counts'],sb['mt']['exc_counts'])
            self.assertEqual(sa['mstd'],sb['mstd'])
        self.assertEqual(calls,list(range(30)))

    def test_frame_partition_does_not_reset_neural_state(self):
        a=VisionModel(2,2,self.settings(frame_ms=20));b=VisionModel(2,2,self.settings(frame_ms=10))
        raw=np.full((3,28,2,2),25.,np.float32)
        one=a.step(raw,0)
        first=b.step(raw,0);second=b.step(raw,.01)
        for key in ('v1_counts','exc_counts','inh_counts'):
            np.testing.assert_array_equal(one['mt'][key],first['mt'][key]+second['mt'][key])
        np.testing.assert_array_equal(a.mt.exc.v,b.mt.exc.v)
        np.testing.assert_array_equal(a.mstd.activity,b.mstd.activity)

    def test_feedback_synaptic_events_arrive_and_decay_after_delay(self):
        model=VisionModel(2,2,self.settings(frame_ms=1))
        model.set_feedback(lambda ctx:dict(feedback_inh_events=np.ones((8,2,2))) if ctx['tick']==0 else None)
        raw=np.zeros((3,28,2,2),np.float32)
        model.step(raw,0);self.assertEqual(model.mt.feedback_gi.g.sum(),0)
        model.step(raw,.001);np.testing.assert_array_equal(model.mt.feedback_gi.g,np.ones((8,2,2)))
        model.step(raw,.002);np.testing.assert_allclose(model.mt.feedback_gi.g,exp(-.1),atol=1e-6)

    def test_actual_mstd_state_can_drive_closed_loop_and_disable_restores_baseline(self):
        with patch('mstd.model.load_projection',return_value=projection()):
            a=VisionModel(2,2,self.settings(frame_ms=10))
            b=VisionModel(2,2,self.settings(frame_ms=10))
        modulation=[]
        def controller(ctx):
            gain=20*max(ctx['mstd']['activity'])
            modulation.append(gain)
            return dict(exc_current=np.full((8,2,2),gain))
        b.set_feedback(controller)
        raw=np.zeros((3,28,2,2),np.float32);raw[0,0]=200
        for t in range(10):a.step(raw,t*.01);b.step(raw,t*.01)
        self.assertGreater(max(modulation),0)
        self.assertFalse(np.array_equal(a.mt.exc.v,b.mt.exc.v))
        b.set_feedback(None);a.reset();b.reset()
        for t in range(5):
            sa=a.step(raw,t*.01);sb=b.step(raw,t*.01)
            np.testing.assert_array_equal(sa['mt']['exc_counts'],sb['mt']['exc_counts'])
            self.assertEqual(sa['mstd'],sb['mstd'])

    def test_downstream_neuron_can_receive_mt_spikes_without_rate_or_file_bridge(self):
        with patch('mstd.model.load_projection',return_value=projection()):
            model=VisionModel(2,2,self.settings(frame_ms=10))
        op=model.ops
        class NeuralReceiver:
            def __init__(self):
                self.delay=DelayLine(op,(1,));self.syn=ExpConductance(op,(1,),5)
                self.neuron=IzhikevichPopulation(op,(1,));self.received=0;self.emitted=0;self.calls=0
            def step(self,signal,duration_s):
                dt=duration_s*1000
                g=self.syn.step(self.delay.read(),dt)
                spikes=self.neuron.step(g,np.zeros(1,np.float32),dt)
                amount=float(signal['spikes'].sum());self.received+=amount
                self.delay.write(np.array([amount],np.float32))
                self.emitted+=float(spikes.sum());self.calls+=1
                return dict(spikes=spikes)
        receiver=NeuralReceiver();model.mstd=receiver
        raw=np.zeros((3,28,2,2),np.float32);raw[0,0]=200
        for t in range(10):model.step(raw,t*.01)
        self.assertEqual(receiver.calls,100)
        self.assertGreater(receiver.received,0);self.assertGreater(receiver.emitted,0)

    def test_monitor_files_passivity_event_counts_and_v1_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);stim=root/'stim.dat'
            write_stimulus(stim,19,19,stimulus_frames(19,19,4,0,1.5))
            base=dict(mt_backend='python_spiking',mt_radius=0,projection_config=PROJECTION,
                      input=dict(path=str(stim)),progress_every=0)
            on=simulate({**base,'output':str(root/'on'),'monitors':dict(spikes=True,neural_state=True)})
            off={k:False for k in ('v1','mt_projection','mt_pooling','mt_normalization','mstd','summary','spikes','neural_state')}
            silent=simulate({**base,'output':str(root/'off'),'monitors':off})
            self.assertEqual(on['last_mstd'],silent['last_mstd'])
            self.assertFalse((root/'off/monitors').exists())
            replay=simulate({**base,'input':dict(kind='v1_recording',path=str(root/'on/monitors/v1')),
                             'layers':dict(v1=False),'output':str(root/'replay')})
            self.assertEqual(on['last_mstd'],replay['last_mstd'])
            dtype=np.dtype([('tick','<u8'),('neuron','<u4'),('count','<u4')])
            events=np.fromfile(root/'on/monitors/spikes/exc.events',dtype=dtype)
            observed=np.fromfile(root/'on/monitors/mt_exc_rates/values.f32',dtype='<f4').reshape(4,8,19,19)
            reconstructed=np.zeros_like(observed)
            for row in events:
                reconstructed[int(row['tick'])//50].reshape(-1)[int(row['neuron'])]+=int(row['count'])*20
            np.testing.assert_array_equal(reconstructed,observed)
            meta=json.loads((root/'on/monitors/spikes/metadata.json').read_text())
            self.assertEqual(meta['ticks'],200)
            voltage=json.loads((root/'on/monitors/mt_exc_voltage/metadata.json').read_text())
            self.assertEqual(voltage['timestamps_s'],[.05,.1,.15000000000000002,.2])

    def test_neural_config_and_feedback_validation(self):
        with self.assertRaises(ValueError):model_config(self.settings(frame_ms=1,spiking=dict(dt_ms=.3)))
        model=VisionModel(2,2,self.settings(frame_ms=1))
        model.set_feedback(lambda ctx:dict(exc_current=np.array([np.nan])))
        with self.assertRaises(ValueError):model.step(np.zeros((3,28,2,2)),0)

    @unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch not installed')
    def test_torch_fixed_event_circuit_matches_numpy(self):
        cpu=PythonSpikingMT(ArrayOps(),projection(),2,2)
        torch=PythonSpikingMT(ArrayOps('torch','cpu'),projection(),2,2)
        raw=np.zeros(cpu.input_shape,np.float32)
        cpu.set_rates(raw);torch.set_rates(torch.op.array(raw))
        for t in range(100):
            events=raw.copy()
            if t%10==0:events[0,0,0,0]=1
            a=cpu.tick(v1_events=events);b=torch.tick(v1_events=torch.op.array(events))
            np.testing.assert_array_equal(a['exc'],torch.op.host(b['exc']))
            np.testing.assert_allclose(cpu.exc.v,torch.op.host(torch.exc.v),atol=1e-3,rtol=1e-5)


if __name__=='__main__':unittest.main()
