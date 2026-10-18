"""Validated, direction-coded V1 input. Raw filter outputs need a separate adapter."""

from dataclasses import dataclass
from math import isfinite, pi


@dataclass(frozen=True)
class V1Spec:
    width: int
    height: int
    directions_rad: tuple[float, ...]
    fx_px: float
    fy_px: float
    cx_px: float
    cy_px: float

    def __post_init__(self):
        if type(self.width) is not int or type(self.height) is not int:
            raise ValueError("width/height must be integers")
        if self.width < 2 or self.height < 2:
            raise ValueError("width/height must be >= 2")
        if len(self.directions_rad) < 4 or not all(isfinite(d) for d in self.directions_rad):
            raise ValueError("at least four finite direction angles required")
        angles = sorted(d % (2 * pi) for d in self.directions_rad)
        if any(abs(a - b) < 1e-7 for a, b in zip(angles, angles[1:])):
            raise ValueError("duplicate direction channels")
        if angles[0] + 2 * pi - angles[-1] < 1e-7:
            raise ValueError("duplicate direction channels")
        gaps = [b-a for a,b in zip(angles, angles[1:])] + [angles[0]+2*pi-angles[-1]]
        if any(abs(gap-2*pi/len(angles)) > 1e-6 for gap in gaps):
            raise ValueError("reference decoder requires uniformly spaced full-circle directions")
        if not all(isfinite(v) for v in (self.fx_px, self.fy_px, self.cx_px, self.cy_px)):
            raise ValueError("finite camera intrinsics required")
        if self.fx_px <= 0 or self.fy_px <= 0:
            raise ValueError("positive focal lengths required")

    @property
    def frame_size(self):
        return self.width * self.height * len(self.directions_rad)


@dataclass(frozen=True)
class V1Frame:
    timestamp_s: float
    rates_hz: tuple[float, ...]

    def validate(self, spec):
        if not isfinite(self.timestamp_s) or self.timestamp_s < 0:
            raise ValueError("timestamp must be finite and nonnegative")
        if len(self.rates_hz) != spec.frame_size:
            raise ValueError("frame size does not match direction-coded V1 specification")
        if any(not isfinite(r) or r < 0 for r in self.rates_hz):
            raise ValueError("rates must be finite and nonnegative Hz")
