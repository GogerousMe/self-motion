"""Deterministic ideal MT fields and diagnostics for MSTd time responses."""
from dataclasses import replace
import numpy as np

from .mstd import DynamicsConfig


def temporal_variants():
    reference = DynamicsConfig()
    return dict(reference=reference,
                input_only=replace(reference, recurrent_gain=0, inhibition=0),
                no_self=replace(reference, recurrent_gain=0),
                no_lateral=replace(reference, inhibition=0),
                tau40=replace(reference, tau_s=.04),
                tau160=replace(reference, tau_s=.16))


def ideal_sequence(spec, case, frames=100, seed=43):
    if type(frames) is not int or frames < 80:
        raise ValueError('at least80 frames required to include all protocol windows')
    if case not in ('hold', 'step', 'sweep', 'occlusion', 'uniform', 'noise_stop'):
        raise ValueError('unsupported temporal case')
    rng = np.random.default_rng(seed)
    y, x = np.indices((spec.height, spec.width))
    for frame in range(frames):
        heading = -7.5
        if case == 'step' and frame >= 40:
            heading = 10.
        if case == 'sweep':
            heading = -12 + 24 * min(max((frame - 20) / 60, 0), 1)
        cx = spec.cx_px + spec.fx_px * np.tan(np.radians(heading))
        vectors = np.stack((x - cx, y - spec.cy_px)).astype(np.float32)
        length = np.linalg.norm(vectors, axis=0)
        vectors = np.divide(vectors, length[None], out=np.zeros_like(vectors), where=length[None] > 0) * .7
        phase = 'clean'
        if case == 'occlusion' and 40 <= frame < 50:
            vectors[:, :, :spec.width // 2] = 0
            phase = 'occlusion'
        if case == 'uniform':
            vectors[0] = .7
            vectors[1] = 0
            heading = None
            phase = 'uniform'
        if case == 'noise_stop':
            if 40 <= frame < 50:
                angles = rng.uniform(-np.pi, np.pi, length.shape)
                mask = rng.random(length.shape) < .5
                noise = np.stack((np.cos(angles), np.sin(angles))) * .7
                vectors[:, mask] = noise[:, mask]
                phase = 'noise'
            if frame >= 60:
                vectors[:] = 0
                heading = None
                phase = 'stopped'
        yield vectors, dict(frame=frame, heading_deg=heading, phase=phase)


def response_metrics(rows, frame_s=.05, tolerance_deg=2.5, case=None):
    stable = [r for r in rows if r['frame'] >= 20]
    targets = [r for r in stable if r['heading_deg'] is not None]
    accepted = [r for r in targets if r['bearing_valid']]
    rejected = [r for r in stable if r['heading_deg'] is None]
    errors = [abs(r['candidate_heading_deg'] - r['heading_deg']) for r in targets]
    result = dict(evaluation_frames=len(stable), target_frames=len(targets),
                  valid_target_frames=len(accepted),
                  target_valid_fraction=len(accepted)/len(targets) if targets else None,
                  candidate_mae_deg=float(np.mean(errors)) if errors else None,
                  accepted_mae_deg=float(np.mean([abs(r['alpha_x_deg']-r['heading_deg']) for r in accepted])) if accepted else None,
                  invalid_target_frames=sum(r['bearing_valid'] for r in rejected),
                  nonheading_frames=len(rejected), tolerance_deg=tolerance_deg)
    for name, start in (('switch', 40), ('recovery', 50)):
        latency = None
        applicable = case == 'step' if name == 'switch' else case in ('occlusion', 'noise_stop')
        for i in range(start, len(rows)-2) if applicable else ():
            window = rows[i:i+3]
            if all(r['bearing_valid'] and r['heading_deg'] is not None and
                   abs(r['alpha_x_deg']-r['heading_deg']) <= tolerance_deg for r in window):
                latency = (i+1-start)*frame_s
                break
        result[name+'_settle_s'] = latency
    residuals = [r['candidate_heading_deg']-r['heading_deg'] for r in targets if r['phase']=='clean']
    result['clean_residual_std_deg'] = float(np.std(residuals)) if residuals else None
    result['final_peak_activity'] = max(rows[-1]['activity'])
    return result
