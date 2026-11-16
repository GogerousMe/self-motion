"""Pixel and ground-truth reflection for controlled horizontal motion pairs."""
from copy import deepcopy

import numpy as np


def reflect_image(image, width, height):
    """Reflect grayscale pixels about x=(width-1)/2, preserving byte order TYX."""
    if len(image) != width*height:
        raise ValueError('expected one width*height grayscale byte frame')
    return np.frombuffer(image, dtype=np.uint8).reshape(height, width)[:, ::-1].tobytes()


def reflect_truth(truth, width):
    """Reflect camera/world x and image u consistently; retain IDs and times.

    Point centers may extend beyond pixel-center boundaries while their soft
    footprints remain visible. Do not clip reflected u or re-spawn points.
    """
    result = deepcopy(truth)
    camera = result['camera']
    for key in ('x_m', 'vx_mps', 'yaw_rad', 'yaw_rate_rad_s',
                'heading_camera_deg', 'signed_curvature_per_m'):
        if camera.get(key) is not None:
            camera[key] = -camera[key]
    for point in result['sparse_flow_to_next_frame']:
        point['u_px'] = width-1-point['u_px']
        if point['du_px'] is not None:
            point['du_px'] = -point['du_px']
    return result


def pair_symmetry(left_headings, right_headings):
    """Summarize matched-frame reflection deviations; not a significance test."""
    left = np.asarray(left_headings, dtype=np.float64)
    right = np.asarray(right_headings, dtype=np.float64)
    if left.ndim!=1 or left.size==0 or left.shape!=right.shape:
        raise ValueError('nonempty matched one-dimensional heading sequences required')
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('finite headings required')
    total = left+right
    return dict(mean_common_bias_deg=float(total.mean()/2),
                mean_absolute_mirror_deviation_deg=float(np.abs(total).mean()))
