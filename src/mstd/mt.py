"""Feedforward direction pooling baseline; no claim of aperture resolution."""

from dataclasses import dataclass
from math import cos, hypot, isfinite, sin


@dataclass(frozen=True)
class MTField:
    vx: tuple[float, ...]
    vy: tuple[float, ...]
    evidence: float


def encode_mt(spec, frame, radius=1):
    frame.validate(spec)
    if type(radius) is not int or radius < 0:
        raise ValueError("pool radius must be a nonnegative integer")
    w, h = spec.width, spec.height
    n = w * h
    # Pool each direction before local divisive normalization. Keep space intact.
    pooled = []
    for d in range(len(spec.directions_rad)):
        plane = []
        for y in range(h):
            for x in range(w):
                values = [frame.rates_hz[d*n+yy*w+xx]
                          for yy in range(max(0, y-radius), min(h, y+radius+1))
                          for xx in range(max(0, x-radius), min(w, x+radius+1))]
                # Stable mean of nonnegative values.
                plane.append(sum(v / len(values) for v in values))
        pooled.append(plane)
    vx, vy = [], []
    directions = [(cos(d), sin(d)) for d in spec.directions_rad]
    for p in range(n):
        maximum = max(plane[p] for plane in pooled)
        # Scale before summing to avoid overflow on large, finite input rates.
        scaled = [plane[p] / maximum if maximum else 0.0 for plane in pooled]
        total = sum(scaled)
        denominator = total + (1.0 / maximum if maximum else 1.0)
        vx.append(sum(a * unit[0] for a, unit in zip(scaled, directions)) / denominator)
        vy.append(sum(a * unit[1] for a, unit in zip(scaled, directions)) / denominator)
    evidence = sum(hypot(x, y) for x, y in zip(vx, vy)) / n
    if not isfinite(evidence):
        raise ValueError("nonfinite MT evidence")
    return MTField(tuple(vx), tuple(vy), evidence)
