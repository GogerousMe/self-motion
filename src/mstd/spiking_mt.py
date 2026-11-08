"""In-memory Python MT circuit: delayed Poisson input, E/I neurons, fast synapses."""
from math import exp
import numpy as np

from .neural import (SpikeConfig,DelayLine,PoissonEncoder,ExpConductance,
                     IzhikevichPopulation,nonnegative,finite)


class PythonSpikingMT:
    def __init__(self,op,projection,width,height,config=None,population_order=('exc','inh')):
        self.op,self.projection=op,projection
        self.config=config or SpikeConfig()
        if len(population_order)!=2 or set(population_order)!={'exc','inh'}:raise ValueError('each MT population must update exactly once')
        self.population_order=tuple(population_order)
        self.shape=(8,height,width)
        self.input_shape=(3,28,height,width)
        w=np.array(projection.weights,np.float32)*projection.gain*self.config.input_weight
        self.positive=op.array(np.maximum(w,0));self.negative=op.array(np.maximum(-w,0))
        self.scales=op.array(projection.scale_weights)[:,None,None,None]
        self.units=op.array([(np.cos(a),np.sin(a)) for a in projection.directions_rad])
        self.encoder=PoissonEncoder(op,self.config.seed)
        self.exc=IzhikevichPopulation(op,self.shape)
        self.inh=IzhikevichPopulation(op,self.shape)
        self.ge=ExpConductance(op,self.shape,self.config.tau_exc_ms)
        self.gi=ExpConductance(op,self.shape,self.config.tau_inh_ms)
        self.drive_i=ExpConductance(op,self.shape,self.config.tau_exc_ms)
        self.feedback_ge=ExpConductance(op,self.shape,self.config.tau_exc_ms)
        self.feedback_gi=ExpConductance(op,self.shape,self.config.tau_inh_ms)
        self.input_delay=DelayLine(op,(2,*self.shape),self.config.delay_ticks)
        self.inhib_delay=DelayLine(op,self.shape,self.config.delay_ticks)
        self.reset()

    def reset(self):
        for part in (self.encoder,self.exc,self.inh,self.ge,self.gi,self.drive_i,self.feedback_ge,self.feedback_gi,self.input_delay,self.inhib_delay):part.reset()
        self.trace=self.op.zeros(self.shape)
        self.zero=self.op.zeros(self.shape)
        self.rates=None
        self.ticks=0

    def set_rates(self,rates):
        if tuple(rates.shape)!=self.input_shape or not nonnegative(self.op,rates):
            raise ValueError('expected finite nonnegative SFYX V1 rates')
        self.rates=rates*self.config.v1_rate_scale
        if not finite(self.op,self.rates):raise ValueError('scaled V1 rates overflow')

    def tick(self,feedback=None,v1_events=None):
        if self.rates is None:raise RuntimeError('set V1 rates before advancing')
        op,cfg=self.op,self.config
        if v1_events is None:events=self.encoder.sample(self.rates,cfg.dt_ms)
        else:
            events=v1_events
            if tuple(events.shape)!=self.input_shape or not nonnegative(op,events) or not bool((events==events.round()).all()):
                raise ValueError('V1 events must be nonnegative integer counts in SFYX layout')
        # Read ALL arrivals before either population is updated.
        arriving=self.input_delay.read()
        inhib=self.inhib_delay.read()*cfg.inhibition_weight
        ge=self.ge.step(arriving[0],cfg.dt_ms)
        gi=self.gi.step(inhib,cfg.dt_ms)
        drive_i=self.drive_i.step(arriving[1],cfg.dt_ms)
        feedback=op.zeros((4,*self.shape)) if feedback is None else feedback
        if tuple(feedback.shape)!=(4,*self.shape) or not finite(op,feedback) or not nonnegative(op,feedback[2:]):
            raise ValueError('invalid MT feedback: currents may be signed, synaptic events must be nonnegative')
        feedback_ge=self.feedback_ge.step(feedback[2],cfg.dt_ms)
        feedback_gi=self.feedback_gi.step(feedback[3],cfg.dt_ms)
        kwargs=dict(dt_ms=cfg.dt_ms,substeps=cfg.substeps,reversal_exc=cfg.reversal_exc,reversal_inh=cfg.reversal_inh)
        inputs={'exc':(self.exc,ge+feedback_ge,gi+feedback_gi,feedback[0]),
                'inh':(self.inh,drive_i,self.zero,feedback[1])}
        emitted={}
        for name in self.population_order:
            population,g_exc,g_inh,external=inputs[name]
            emitted[name]=population.step(g_exc,g_inh,external=external,**kwargs)
        exc,inh=emitted['exc'],emitted['inh']
        weighted=events*self.scales
        scaled=weighted.sum(dim=0) if op.torch else weighted.sum(axis=0)
        einsum=op.torch.einsum if op.torch else np.einsum
        self.input_delay.write(op.stack([einsum('fyx,fd->dyx',scaled,w) for w in (self.positive,self.negative)]))
        self.inhib_delay.write(inh)
        decay=exp(-cfg.dt_ms/cfg.trace_tau_ms)
        self.trace=self.trace*decay+exc*((1-decay)*1000/cfg.dt_ms)
        self.ticks+=1
        return {'v1':events,'exc':exc,'inh':inh}

    def readout(self,host_evidence=True):
        op=self.op
        total=self.trace.sum(dim=0) if op.torch else self.trace.sum(axis=0)
        activity=self.trace/(1+total)
        einsum=op.torch.einsum if op.torch else np.einsum
        vectors=einsum('dyx,dc->cyx',activity,self.units)
        norm=(vectors*vectors).sum(dim=0).sqrt() if op.torch else np.sqrt((vectors*vectors).sum(axis=0))
        evidence=norm.mean()
        return dict(activity=activity,vectors=vectors,evidence=float(evidence) if host_evidence else evidence,pooled_rates=self.trace)

    def diagnostics(self):
        # Copy so external monitors/controllers cannot mutate persistent state.
        return {name:self.op.copy(value) for name,value in dict(
            exc_voltage=self.exc.v,inh_voltage=self.inh.v,exc_recovery=self.exc.u,inh_recovery=self.inh.u,
            exc_conductance=self.ge.g+self.feedback_ge.g,
            inh_conductance=self.gi.g+self.feedback_gi.g).items()}
