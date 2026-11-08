"""Array implementations of the existing projection/MT/MSTd equations.

These are rate-model adapters, not CARLsim E/I spiking dynamics.
"""
from math import atan, degrees, isfinite

import numpy as np

from .mstd import DynamicsConfig


class DirectionMT:
    def __init__(self,ops,projection,width,height,radius=1):
        if type(radius) is not int or radius < 0:
            raise ValueError('nonnegative integer MT radius required')
        self.ops,self.projection,self.radius = ops,projection,radius
        self.shape = (3,28,height,width)
        self.weights = ops.array(projection.weights)
        self.scale_weights = ops.array(projection.scale_weights)[:,None,None,None]
        self.units = ops.array([(np.cos(a),np.sin(a)) for a in projection.directions_rad])
        # Radius >= image dimensions is equivalent to covering the whole axis.
        self.rx,self.ry = min(radius,width-1),min(radius,height-1)
        ones = ops.array(np.ones((height,width),np.float32))
        self.counts = ops.correlate(ops.correlate(ones,[1]*(2*self.rx+1),-1),
                                    [1]*(2*self.ry+1),-2)

    def step(self,raw_rates):
        op = self.ops
        if tuple(raw_rates.shape) != self.shape:
            raise ValueError('expected raw V1 SFYX array')
        if op.torch:
            finite = bool(op.torch.isfinite(raw_rates).all()) and bool((raw_rates>=0).all())
            scaled = (raw_rates*self.scale_weights).sum(dim=0)
            projected = op.torch.einsum('fyx,fd->dyx',scaled,self.weights)
            directions = projected.clamp_min(0)*self.projection.gain
        else:
            finite = np.isfinite(raw_rates).all() and (raw_rates>=0).all()
            scaled = (raw_rates*self.scale_weights).sum(axis=0)
            projected = np.einsum('fyx,fd->dyx',scaled,self.weights,optimize=True)
            directions = np.maximum(projected,0)*self.projection.gain
        if not finite:
            raise ValueError('V1 rates must be finite and nonnegative')
        pooled = op.correlate(op.correlate(directions,[1]*(2*self.rx+1),-1),
                              [1]*(2*self.ry+1),-2)/self.counts
        if op.torch:
            maximum = pooled.amax(dim=0)
            safe = maximum.clamp_min(op.torch.finfo(pooled.dtype).tiny)
            scaled = pooled/safe
            activity = scaled/(scaled.sum(dim=0)+1/safe)
            vectors = op.torch.einsum('dyx,dc->cyx',activity,self.units)
            evidence = float(op.torch.sqrt((vectors*vectors).sum(dim=0)).mean())
        else:
            maximum = pooled.max(axis=0)
            safe = np.maximum(maximum,np.finfo(np.float32).tiny)
            scaled = pooled/safe
            activity = scaled/(scaled.sum(axis=0)+1/safe)
            vectors = np.einsum('dyx,dc->cyx',activity,self.units,optimize=True)
            evidence = float(np.sqrt((vectors*vectors).sum(axis=0)).mean())
        return {'direction_rates':directions,'pooled_rates':pooled,'activity':activity,
                'vectors':vectors,'evidence':evidence}


class TensorMSTd:
    """Same fixed radial templates and Euler update as mstd.MSTd."""
    def __init__(self,ops,spec,centers,config=None):
        self.ops,self.spec,self.centers = ops,spec,tuple(tuple(c) for c in centers)
        if len(self.centers)<2 or len(set(self.centers))!=len(self.centers):
            raise ValueError('at least two distinct centers required')
        self.config = config or DynamicsConfig()
        y,x = np.indices((spec.height,spec.width),dtype=np.float64)
        patterns = []
        for cx,cy in self.centers:
            if not all(isfinite(v) for v in (cx,cy)):
                raise ValueError('finite centers required')
            dx,dy = x-cx,y-cy
            length = np.hypot(dx,dy)
            patterns.append(np.stack((np.divide(dx,length,out=np.zeros_like(dx),where=length>0),
                                      np.divide(dy,length,out=np.zeros_like(dy),where=length>0))))
        self.templates = ops.array(np.stack(patterns))
        self.reset()

    def reset(self):
        self.activity = self.ops.zeros((len(self.centers),))

    def step(self,mt,duration_s,return_state=True):
        if not isfinite(duration_s) or duration_s<=0:
            raise ValueError('positive finite duration required')
        op,cfg = self.ops,self.config
        v = mt['vectors']
        if op.torch:
            norm = (v*v).sum(dim=0).sqrt().sum()
            score = (self.templates*v).sum(dim=(1,2,3))/norm.clamp_min(1e-12)
            score = score.clamp_min(0)
            score = op.torch.where(norm>1e-12,score,op.zeros((len(self.centers),)))
        else:
            norm = np.sqrt((v*v).sum(axis=0)).sum()
            score = np.maximum((self.templates*v).sum(axis=(1,2,3))/max(float(norm),1e-12),0)
            if norm<=1e-12:
                score = op.zeros((len(self.centers),))
        remaining = duration_s
        while remaining>1e-12:
            dt = min(cfg.max_step_s,remaining)
            a = self.activity
            feedback = a*a/(0.25+a*a)
            inhibition = cfg.inhibition*(feedback.sum()-feedback)/(len(self.centers)-1)
            derivative = (-a+(1-a)*(cfg.gain*score+cfg.recurrent_gain*feedback)-a*inhibition)/cfg.tau_s
            self.activity = (a+dt*derivative).clamp(0,1) if op.torch else np.clip(a+dt*derivative,0,1)
            remaining -= dt
        if not return_state:return None
        activities = op.host(self.activity).tolist()
        order = sorted(range(len(activities)),key=activities.__getitem__,reverse=True)
        winner,runner = order[:2]
        margin = activities[winner]-activities[runner]
        valid = mt['evidence']>=cfg.min_evidence and margin>=cfg.min_margin
        cx,cy = self.centers[winner]
        calibrated = all(getattr(self.spec,k,None) is not None for k in ('fx_px','fy_px','cx_px','cy_px'))
        bearing_valid = valid and calibrated
        return {'valid':valid,'bearing_valid':bearing_valid,'winner_center_px':[cx,cy],
                'alpha_x_deg':degrees(atan((cx-self.spec.cx_px)/self.spec.fx_px)) if bearing_valid else None,
                'alpha_y_deg':degrees(atan((cy-self.spec.cy_px)/self.spec.fy_px)) if bearing_valid else None,
                'winner_index':winner,'margin':margin,'mt_evidence':mt['evidence'],
                'activity':activities,'template_scores':op.host(score).tolist()}
