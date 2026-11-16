"""Optional local angular readout of the existing horizontal MSTd activities."""
from dataclasses import dataclass
from math import atan, degrees, isclose, isfinite


@dataclass(frozen=True)
class ReadoutConfig:
    mode: str = 'winner'
    curvature_epsilon: float = 1e-12

    def __post_init__(self):
        if self.mode not in ('winner','local_quadratic'):
            raise ValueError('unsupported MSTd readout mode')
        value=self.curvature_epsilon
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not isfinite(value) or value<=0:
            raise ValueError('positive finite curvature_epsilon required')


def heading_readout(activity, centers, spec, winner, valid, config=None):
    """Select a bearing without modifying winner, activities or acceptance.

    Interpolation is supported only on an ascending, equally spaced angular
    horizontal grid. Pixel coordinates are not treated as equally spaced angles.
    All published bearings remain null when invalid or without intrinsics.
    """
    cfg=config or ReadoutConfig()
    calibrated=all(getattr(spec,k,None) is not None for k in ('fx_px','fy_px','cx_px','cy_px'))
    detail=dict(mode=cfg.mode,curvature_epsilon=cfg.curvature_epsilon,
                method_used='winner',fallback_reason=None,offset_bins=0.)
    empty=dict(bearing_valid=False,alpha_x_deg=None,alpha_y_deg=None,
               winner_alpha_x_deg=None,local_quadratic_alpha_x_deg=None,readout=detail)
    if not calibrated:
        detail.update(method_used=None,fallback_reason='uncalibrated')
        return empty
    if not valid:
        detail.update(method_used=None,fallback_reason='invalid_input')
        return empty
    if len(activity)!=len(centers) or not 0<=winner<len(activity) or not all(isfinite(a) for a in activity):
        raise ValueError('finite matching activity and a valid winner required')
    activity=[float(a) for a in activity]
    angles=[degrees(atan((cx-spec.cx_px)/spec.fx_px)) for cx,cy in centers]
    base=angles[winner];selected=base;quadratic=None
    if cfg.mode=='local_quadratic':
        quadratic=base
        if len(centers)<3:
            detail['fallback_reason']='too_few_templates'
        elif not all(isclose(c[1],centers[0][1],abs_tol=1e-9,rel_tol=0) for c in centers):
            detail['fallback_reason']='non_horizontal_grid'
        elif angles[1]<=angles[0] or not all(isclose(angles[i+1]-angles[i],angles[1]-angles[0],abs_tol=1e-9,rel_tol=1e-9)
                                             for i in range(len(angles)-1)):
            detail['fallback_reason']='non_uniform_angular_grid'
        elif winner==0 or winner==len(centers)-1:
            detail['fallback_reason']='boundary_peak'
        else:
            left,mid,right=activity[winner-1:winner+2]
            curvature=left-2*mid+right
            if curvature>=-cfg.curvature_epsilon:
                detail['fallback_reason']='flat_or_nonconcave_peak'
            else:
                offset=max(-.5,min(.5,.5*(left-right)/curvature))
                quadratic=base+offset*(angles[1]-angles[0])
                detail.update(method_used='local_quadratic',offset_bins=offset)
        selected=quadratic
    return dict(bearing_valid=True,alpha_x_deg=selected,
                alpha_y_deg=degrees(atan((centers[winner][1]-spec.cy_px)/spec.fy_px)),
                winner_alpha_x_deg=base,local_quadratic_alpha_x_deg=quadratic,readout=detail)
