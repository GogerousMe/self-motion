"""Direction-specific spatial excitation in effective conductance units."""
from dataclasses import asdict, dataclass
from math import isfinite

import numpy as np

from .neural import DelayLine, ExpConductance, nonnegative


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
