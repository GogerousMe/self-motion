"""Small array-native neural primitives. Time is ms; external rates are Hz.

This is a project model, not a CARLsim compatibility implementation. The same
population and delay components can be reused by later spiking MSTd circuits.
"""
from dataclasses import dataclass, asdict
from math import exp, isfinite
import numpy as np


def finite(op, value):
    return bool(op.torch.isfinite(value).all()) if op.torch else bool(np.isfinite(value).all())


def nonnegative(op,value):
    return finite(op,value) and bool((value>=0).all())


def where(op,condition,yes,no):
    return op.torch.where(condition,yes,no) if op.torch else np.where(condition,yes,no)


@dataclass(frozen=True)
class SpikeConfig:
    dt_ms: float = 1.0
    substeps: int = 4
    seed: int = 42
    v1_rate_scale: float = 1.0
    input_weight: float = 1.5
    inhibition_weight: float = 1.0
    tau_exc_ms: float = 5.0
    tau_inh_ms: float = 10.0
    reversal_exc: float = 0.0
    reversal_inh: float = -80.0
    trace_tau_ms: float = 20.0
    delay_ticks: int = 1
    feedback_delay_ticks: int = 1

    def __post_init__(self):
        for key in ('dt_ms','v1_rate_scale','tau_exc_ms','tau_inh_ms','trace_tau_ms'):
            value=getattr(self,key)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not isfinite(value) or value<=0:
                raise ValueError(f'positive finite {key} required')
        for key in ('input_weight','inhibition_weight'):
            value=getattr(self,key)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not isfinite(value) or value<0:
                raise ValueError(f'nonnegative finite {key} required')
        if any(not isfinite(getattr(self,k)) for k in ('reversal_exc','reversal_inh')):
            raise ValueError('finite reversal potentials required')
        for key in ('substeps','delay_ticks','feedback_delay_ticks'):
            if type(getattr(self,key)) is not int or getattr(self,key)<1:raise ValueError(f'positive integer {key} required')
        if type(self.seed) is not int or not 0<=self.seed<2**63:raise ValueError('invalid seed')
        if self.dt_ms/self.substeps>0.5:raise ValueError('neuron integration substep must be <=0.5ms')

    def description(self):return asdict(self)


class DelayLine:
    """Read before updating populations, write new events afterwards.

    With delay=1, events emitted at tick k's END arrive at tick k+1's START.
    Values read here are owned by the caller; writes copy their input.
    """
    def __init__(self,op,shape,delay=1):
        if type(delay) is not int or delay<1:raise ValueError('positive integer delay required')
        self.op,self.shape,self.delay=op,tuple(shape),delay
        self.reset()

    def reset(self):
        self.buffer=[self.op.zeros(self.shape) for _ in range(self.delay)]
        self.index=0

    def read(self):return self.op.copy(self.buffer[self.index])

    def write(self,value):
        if tuple(value.shape)!=self.shape:raise ValueError('delay shape mismatch')
        self.buffer[self.index]=self.op.copy(value)
        self.index=(self.index+1)%self.delay


class PoissonEncoder:
    """Exact Poisson COUNTS per time bin; multiple events in a bin are retained.

    Events are assigned to the bin end. This is not CARLsim's refractory input
    generator. CPU and torch use independent RNG algorithms, not identical draws.
    """
    def __init__(self,op,seed):self.op,self.seed=op,seed;self.reset()

    def reset(self):
        if self.op.torch:
            self.rng=self.op.torch.Generator(device=self.op.device).manual_seed(self.seed)
        else:self.rng=np.random.default_rng(self.seed)

    def sample(self,rates,dt_ms):
        if not isfinite(dt_ms) or dt_ms<=0 or not nonnegative(self.op,rates):
            raise ValueError('finite nonnegative rates and positive dt required')
        mean=rates*(dt_ms/1000)
        return self.op.torch.poisson(mean,generator=self.rng) if self.op.torch else self.rng.poisson(mean).astype(np.float32)


class ExpConductance:
    """Aggregate fixed-weight synapses sharing a decay constant, per receiver."""
    def __init__(self,op,shape,tau_ms):
        if not isfinite(tau_ms) or tau_ms<=0:raise ValueError('positive tau required')
        self.op,self.shape,self.tau_ms=op,tuple(shape),tau_ms
        self.reset()

    def reset(self):self.g=self.op.zeros(self.shape)

    def step(self,arrival,dt_ms):
        if tuple(arrival.shape)!=self.shape or not nonnegative(self.op,arrival) or not isfinite(dt_ms) or dt_ms<=0:
            raise ValueError('invalid conductance input')
        self.g=self.g*exp(-dt_ms/self.tau_ms)+arrival
        return self.g


class IzhikevichPopulation:
    """Vectorized v/u neurons; simultaneous Euler v/u update within each substep.

    The conventional Izhikevich numerical units are used (v in mV, time in ms).
    Conductances are effective model units; they are not calibrated biological nS.
    No extra refractory rule, slow receptors, plasticity, or hidden voltage clip.
    """
    def __init__(self,op,shape,a=.02,b=.2,c=-65.,d=8.):
        if not all(isfinite(v) for v in (a,b,c,d)) or a<=0:raise ValueError('invalid neuron parameters')
        self.op,self.shape=op,tuple(shape)
        self.a,self.b,self.c,self.d=a,b,c,d
        self.reset()

    def reset(self):
        self.v=self.op.zeros(self.shape)+self.c
        self.u=self.v*self.b

    def step(self,g_exc,g_inh,dt_ms,substeps=4,external=None,reversal_exc=0.,reversal_inh=-80.):
        if not isfinite(dt_ms) or dt_ms<=0 or type(substeps) is not int or substeps<1 or dt_ms/substeps>0.5:
            raise ValueError('invalid integration step')
        for g in (g_exc,g_inh):
            if tuple(g.shape)!=self.shape or not nonnegative(self.op,g):raise ValueError('invalid neural conductance')
        external=self.op.zeros(self.shape) if external is None else external
        if tuple(external.shape)!=self.shape or not finite(self.op,external):raise ValueError('invalid external current')
        spikes=self.op.zeros(self.shape)
        h=dt_ms/substeps
        for _ in range(substeps):
            v,u=self.v,self.u
            current=g_exc*(reversal_exc-v)+g_inh*(reversal_inh-v)+external
            next_v=v+h*(.04*v*v+5*v+140-u+current)
            next_u=u+h*self.a*(self.b*v-u)
            if not finite(self.op,next_v) or not finite(self.op,next_u):
                raise FloatingPointError('integration overflow before spike reset; review drive or timestep')
            fired=next_v>=30
            self.v=where(self.op,fired,self.c,next_v)
            self.u=next_u+self.op.float32(fired)*self.d
            spikes+=self.op.float32(fired)
        if not finite(self.op,self.v) or not finite(self.op,self.u):
            raise FloatingPointError('nonfinite neuron state; reduce dt/substep or review drive')
        return spikes
