"""Radial-template reference dynamics, inspired by CD, not a paper reproduction."""

from dataclasses import dataclass
from math import atan, degrees, hypot, isfinite, radians, tan


@dataclass(frozen=True)
class DynamicsConfig:
    tau_s: float = 0.08
    max_step_s: float = 0.005
    gain: float = 2.0
    recurrent_gain: float = 0.5
    inhibition: float = 1.0
    min_evidence: float = 0.02
    min_margin: float = 0.001

    def __post_init__(self):
        values = tuple(self.__dict__.values())
        if any(not isfinite(v) or v < 0 for v in values):
            raise ValueError("dynamics parameters must be finite and nonnegative")
        if self.tau_s <= 0 or not 0 < self.max_step_s <= self.tau_s / 4:
            raise ValueError("integration step must be positive and <= tau/4")


class MSTd:
    def __init__(self, spec, centers_px, config=None):
        self.spec = spec
        self.centers = tuple(tuple(c) for c in centers_px)
        if len(self.centers) < 2 or len(set(self.centers)) != len(self.centers):
            raise ValueError("at least two distinct template centers required")
        if any(len(c) != 2 or not all(isfinite(v) for v in c) for c in self.centers):
            raise ValueError("template centers must be finite 2D points")
        self.config = config or DynamicsConfig()
        self.templates = []
        for cx, cy in self.centers:
            vectors = []
            for y in range(spec.height):
                for x in range(spec.width):
                    dx, dy = x-cx, y-cy
                    length = hypot(dx, dy)
                    vectors.append((dx/length, dy/length) if length else (0.0, 0.0))
            self.templates.append(tuple(vectors))
        self.reset()

    def reset(self):
        self.activity = [0.0] * len(self.centers)
        self.elapsed_s = 0.0

    def step(self, mt, duration_s):
        if not isfinite(duration_s) or duration_s <= 0:
            raise ValueError("positive finite duration required")
        n = self.spec.width * self.spec.height
        if len(mt.vx) != n or len(mt.vy) != n:
            raise ValueError("MT field dimensions do not match templates")
        if any(not isfinite(v) for v in (*mt.vx, *mt.vy, mt.evidence)):
            raise ValueError("finite MT field required")
        norm = sum(hypot(x, y) for x, y in zip(mt.vx, mt.vy))
        scores = [max(0.0, sum(x*tx+y*ty for x,y,(tx,ty) in zip(mt.vx, mt.vy, template)) / norm)
                  if norm > 1e-12 else 0.0 for template in self.templates]
        cfg = self.config
        remaining = duration_s
        while remaining > 1e-12:
            dt = min(cfg.max_step_s, remaining)
            feedback = [a*a / (0.25+a*a) for a in self.activity]
            total = sum(feedback)
            updated = []
            for a, drive, f in zip(self.activity, scores, feedback):
                inhibition = cfg.inhibition * (total-f) / (len(feedback)-1)
                derivative = (-a + (1-a)*(cfg.gain*drive+cfg.recurrent_gain*f) - a*inhibition) / cfg.tau_s
                updated.append(max(0.0, min(1.0, a+dt*derivative)))
            self.activity = updated
            remaining -= dt
        self.elapsed_s += duration_s
        order = sorted(range(len(self.activity)), key=self.activity.__getitem__, reverse=True)
        winner, runner = order[:2]
        margin = self.activity[winner] - self.activity[runner]
        valid = mt.evidence >= cfg.min_evidence and margin >= cfg.min_margin
        cx, cy = self.centers[winner]
        # alpha_x/y are projected bearing angles, not Euler yaw/pitch or world pose.
        return {"valid": valid,
                "alpha_x_deg": degrees(atan((cx-self.spec.cx_px)/self.spec.fx_px)) if valid else None,
                "alpha_y_deg": degrees(atan((cy-self.spec.cy_px)/self.spec.fy_px)) if valid else None,
                "winner_index": winner, "margin": margin, "mt_evidence": mt.evidence,
                "activity": list(self.activity), "template_scores": scores}


def horizontal_centers(spec, angles_deg):
    return [(spec.cx_px + spec.fx_px*tan(radians(a)), spec.cy_px) for a in angles_deg]
