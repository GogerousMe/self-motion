"""Direction-specific spatial excitation in effective conductance units."""
from dataclasses import asdict, dataclass
from math import isfinite

import numpy as np

from .neural import DelayLine, ExpConductance, IzhikevichPopulation, nonnegative


COMPETITION_NEURON_PARAMETERS = dict(a=0.1, b=0.2, c=-65.0, d=2.0)


@dataclass(frozen=True)
class CompetitionConfig:
    enabled: bool = False
    mode: str = 'angular'
    input_weight: float = 0.25
    inhibition_weight: float = 0.2
    tau_exc_ms: float = 5.0
    tau_inh_ms: float = 10.0
    input_delay_ticks: int = 1
    output_delay_ticks: int = 1

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError('mt_competition.enabled must be boolean')
        if self.mode not in ('angular', 'opponent'):
            raise ValueError('mt_competition.mode must be angular or opponent')
        for key in ('input_weight', 'inhibition_weight', 'tau_exc_ms', 'tau_inh_ms'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
                raise ValueError(f'finite numeric mt_competition.{key} required')
            if value < 0 or (key.startswith('tau_') and value == 0):
                raise ValueError(f'invalid mt_competition.{key}')
        for key in ('input_delay_ticks', 'output_delay_ticks'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f'positive integer mt_competition.{key} required')

    def description(self):
        return asdict(self)


def direction_inhibition_matrix(directions_rad, mode='angular'):
    """Rows receive inhibition, columns identify the source J direction."""
    angles = np.asarray(directions_rad, dtype=np.float64)
    if angles.shape != (8,) or not np.isfinite(angles).all():
        raise ValueError('eight finite competition directions required')
    distance = abs((angles[:, None] - angles[None, :] + np.pi) % (2 * np.pi) - np.pi)
    if np.any(distance[~np.eye(8, dtype=bool)] < 1e-6):
        raise ValueError('competition directions must be distinct modulo 2pi')
    if mode == 'angular':
        weights = np.maximum(1 - np.cos(distance), 0)
    elif mode == 'opponent':
        weights = (abs(distance - np.pi) < 1e-6).astype(np.float64)
        if not np.all(weights.sum(axis=1) == 1):
            raise ValueError('opponent competition requires one opposite direction per channel')
    else:
        raise ValueError('unsupported competition mode')
    np.fill_diagonal(weights, 0)
    return (weights / weights.sum(axis=1, keepdims=True)).astype(np.float32)


class DirectionCompetition:
    """Local E_d -> delayed J_d -> delayed cross-direction E inhibition.

    Arrivals are read before populations advance. Both new E and J outputs
    are emitted at tick end, with separate positive delays on the two legs.
    """
    def __init__(self, op, directions_rad, width, height, config):
        self.op, self.config = op, config
        self.shape = (8, height, width)
        self.weights = op.array(direction_inhibition_matrix(directions_rad, config.mode))
        self.population = IzhikevichPopulation(op, self.shape, **COMPETITION_NEURON_PARAMETERS)
        self.exc_synapse = ExpConductance(op, self.shape, config.tau_exc_ms)
        self.inh_synapse = ExpConductance(op, self.shape, config.tau_inh_ms)
        self.input_delay = DelayLine(op, self.shape, config.input_delay_ticks)
        self.output_delay = DelayLine(op, self.shape, config.output_delay_ticks)

    def reset(self):
        for part in (self.population, self.exc_synapse, self.inh_synapse,
                     self.input_delay, self.output_delay):
            part.reset()

    def arrive(self, dt_ms):
        excitation = self.exc_synapse.step(self.input_delay.read(), dt_ms)
        inhibition = self.inh_synapse.step(self.output_delay.read(), dt_ms)
        return excitation, inhibition

    def emit(self, exc_spikes, competition_spikes):
        if (tuple(exc_spikes.shape) != self.shape or tuple(competition_spikes.shape) != self.shape
                or not nonnegative(self.op, exc_spikes) or not nonnegative(self.op, competition_spikes)):
            raise ValueError('expected finite nonnegative DYX competition events')
        einsum = self.op.torch.einsum if self.op.torch else np.einsum
        self.input_delay.write(exc_spikes * self.config.input_weight)
        self.output_delay.write(einsum('ds,syx->dyx', self.weights, competition_spikes)
                                * self.config.inhibition_weight)


@dataclass(frozen=True)
class SpatialConfig:
    enabled: bool = False
    radius: int = 2
    sigma_parallel_px: float = 1.5
    sigma_perpendicular_px: float = 0.75
    gain: float = 0.05
    tau_ms: float = 5.0
    delay_ticks: int = 1

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError('mt_spatial.enabled must be boolean')
        for key in ('radius', 'delay_ticks'):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f'positive integer mt_spatial.{key} required')
        for key in ('sigma_parallel_px', 'sigma_perpendicular_px', 'tau_ms', 'gain'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
                raise ValueError(f'finite numeric mt_spatial.{key} required')
            if value < 0 or (key != 'gain' and value == 0):
                raise ValueError(f'invalid mt_spatial.{key}')

    def description(self):
        return asdict(self)


class DirectionSpatialKernel:
    """Same-direction elliptical Gaussian pooling, with receiver normalization.

    Coordinates are x-right, y-down. The center is included. Missing neighbors
    contribute neither activity nor normalization weight. Each channel uses
    the finite image domain, including when the kernel exceeds the grid.
    """
    def __init__(self, op, directions_rad, width, height, config):
        if any(type(size) is not int or size < 1 for size in (width, height)):
            raise ValueError('positive integer spatial grid dimensions required')
        angles = np.asarray(directions_rad, dtype=np.float64)
        if angles.shape != (8,) or not np.isfinite(angles).all():
            raise ValueError('eight finite spatial directions required')
        self.op, self.config = op, config
        self.shape = (8, height, width)
        radius = config.radius
        y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
        cosine = np.cos(angles)[:, None, None]
        sine = np.sin(angles)[:, None, None]
        along = x * cosine + y * sine
        across = -x * sine + y * cosine
        weights = np.exp(-0.5 * ((along / config.sigma_parallel_px) ** 2
                                 + (across / config.sigma_perpendicular_px) ** 2))
        weights /= weights.sum(axis=(1, 2), keepdims=True)
        self.weights = op.array(weights)
        self.support = self._correlate(op.zeros(self.shape) + 1)

    def _correlate(self, values):
        radius = self.config.radius
        if self.op.torch:
            import torch.nn.functional as functional
            return functional.conv2d(values[None], self.weights[:, None],
                                     padding=radius, groups=8)[0]
        height, width = self.shape[1:]
        padded = np.pad(values, ((0, 0), (radius, radius), (radius, radius)))
        result = np.zeros_like(values)
        for y in range(2 * radius + 1):
            for x in range(2 * radius + 1):
                result += self.weights[:, y, x, None, None] * padded[:, y:y + height, x:x + width]
        return result

    def apply(self, values):
        if tuple(values.shape) != self.shape or not nonnegative(self.op, values):
            raise ValueError('expected finite nonnegative DYX spatial activity')
        return self._correlate(values) / self.support


class SpatialExcitation:
    """Read delayed arrivals before neurons update, emit after all populations."""
    def __init__(self, op, directions_rad, width, height, config):
        self.config = config
        self.kernel = DirectionSpatialKernel(op, directions_rad, width, height, config)
        self.delay = DelayLine(op, self.kernel.shape, config.delay_ticks)
        self.synapse = ExpConductance(op, self.kernel.shape, config.tau_ms)

    def reset(self):
        self.delay.reset()
        self.synapse.reset()

    def arrive(self, dt_ms):
        return self.synapse.step(self.delay.read(), dt_ms)

    def emit(self, spikes):
        self.delay.write(self.kernel.apply(spikes) * self.config.gain)
