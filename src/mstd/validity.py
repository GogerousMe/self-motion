"""Optional readout gate for the current in-view radial expansion templates.

This changes acceptance only, never template drives or competition dynamics.
Uniform-flow support is an alternative explanation, not a motion-speed estimate.
"""
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class ValidityConfig:
    mode: str = 'legacy_margin'
    min_template_score: float = 0.2
    min_geometry_excess: float = 0.1
    local_evidence_threshold: float = 0.02
    min_active_fraction: float = 0.25

    def __post_init__(self):
        if self.mode not in ('legacy_margin', 'geometry_support'):
            raise ValueError('unsupported MSTd validity mode')
        for name in ('min_template_score', 'min_geometry_excess',
                     'local_evidence_threshold', 'min_active_fraction'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
                raise ValueError(f'{name} must be finite numeric')
            if not 0 <= value <= 1:
                raise ValueError(f'{name} must be between zero and one')
        if self.local_evidence_threshold <= 0:
            raise ValueError('local_evidence_threshold must be positive')


def decide_validity(config, dynamics, evidence, margin, scores, winner,
                    uniform_score=None, active_fraction=None):
    """Shared scalar decision, used by list, NumPy, torch and saved-state checks.

Geometric support belongs to the CURRENT competition winner, not an unrelated
instantaneous best candidate. Ratios are dimensionless, not probabilities.
"""
    legacy = bool(evidence >= dynamics.min_evidence and margin >= dynamics.min_margin)
    detail = {'mode': config.mode, 'legacy_valid': legacy}
    if config.mode == 'legacy_margin':
        return legacy, detail
    values = (evidence, margin, uniform_score, active_fraction, *scores)
    if any(v is None or not isfinite(v) for v in values):
        raise ValueError('finite geometric validity inputs required')
    selected = float(scores[winner])
    excess = selected - uniform_score
    checks = {
        'evidence': evidence >= dynamics.min_evidence,
        'active_fraction': active_fraction >= config.min_active_fraction,
        'template_score': selected >= config.min_template_score,
        'geometry_excess': excess >= config.min_geometry_excess,
    }
    detail.update(winner_template_score=selected, max_template_score=float(max(scores)),
                  uniform_score=float(uniform_score), geometry_excess=float(excess),
                  active_fraction=float(active_fraction),
                  rejected_by=[k for k, passed in checks.items() if not passed])
    return all(checks.values()), detail
