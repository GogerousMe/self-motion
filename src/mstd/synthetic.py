"""Synthetic direction responses, not video-derived biological V1 responses."""

from math import atan2, cos, exp, pi
from .data import V1Frame, V1Spec
from .mstd import horizontal_centers


def default_spec():
    return V1Spec(16, 16, tuple(i*pi/4 for i in range(8)), 16.0, 16.0, 7.5, 7.5)


def radial_frame(spec, timestamp_s, angle_deg=0.0, patch=None):
    """patch=(x0,y0,x1,y1,direction_rad) replaces a region's local motion."""
    cx, cy = horizontal_centers(spec, [angle_deg])[0]
    rates = []
    for preferred in spec.directions_rad:
        for y in range(spec.height):
            for x in range(spec.width):
                direction = atan2(y-cy, x-cx)
                if patch and patch[0] <= x < patch[2] and patch[1] <= y < patch[3]:
                    direction = patch[4]
                rates.append(30.0 * exp(3.0*(cos(direction-preferred)-1.0)))
    return V1Frame(timestamp_s, tuple(rates))
