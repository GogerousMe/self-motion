"""Camera-frame translation references from camera-to-world poses, for audit."""
import numpy as np


def quaternion_xyzw_rotations(quaternions):
    q=np.asarray(quaternions,dtype=np.float64)
    if q.ndim!=2 or q.shape[1]!=4 or not np.isfinite(q).all():
        raise ValueError('finite Nx4 xyzw quaternions required')
    norm=np.linalg.norm(q,axis=1)
    if (norm<1e-12).any():raise ValueError('nonzero quaternion required')
    x,y,z,w=(q/norm[:,None]).T
    r=np.empty((len(q),3,3))
    r[:,0,0]=1-2*(y*y+z*z);r[:,0,1]=2*(x*y-z*w);r[:,0,2]=2*(x*z+y*w)
    r[:,1,0]=2*(x*y+z*w);r[:,1,1]=1-2*(x*x+z*z);r[:,1,2]=2*(y*z-x*w)
    r[:,2,0]=2*(x*z-y*w);r[:,2,1]=2*(y*z+x*w);r[:,2,2]=1-2*(x*x+y*y)
    return r


def camera_motion_reference(rotations,positions,times,half_window):
    """Use R_i.T*(p_hi-p_lo)/dt; heading is translation bearing, not yaw.

    Returns smoothed displacement-direction references and both step/window
    rotation rates. Centered future ground truth is for evaluation only.
    """
    r=np.asarray(rotations,dtype=np.float64);p=np.asarray(positions,dtype=np.float64);t=np.asarray(times,dtype=np.float64)
    n=len(t)
    if n<2 or r.shape!=(n,3,3) or p.shape!=(n,3) or not all(np.isfinite(a).all() for a in (r,p,t)):
        raise ValueError('finite matched pose/time arrays required')
    if (np.diff(t)<=0).any() or type(half_window) is not int or half_window<1:
        raise ValueError('strictly increasing times and positive integer half_window required')
    lo=np.maximum(np.arange(n)-half_window,0);hi=np.minimum(np.arange(n)+half_window,n-1)
    dt=t[hi]-t[lo]
    velocity=np.einsum('nji,nj->ni',r,p[hi]-p[lo])/dt[:,None]
    def rotation_angle(left,right):
        relative=np.einsum('nji,njk->nik',left,right)
        cosine=np.clip((np.trace(relative,axis1=1,axis2=2)-1)/2,-1,1)
        return np.degrees(np.arccos(cosine))
    step=rotation_angle(r[:-1],r[1:])/np.diff(t)
    return dict(velocity_camera_mps=velocity,speed_mps=np.linalg.norm(velocity,axis=1),
        heading_x_deg=np.degrees(np.arctan2(velocity[:,0],velocity[:,2])),
        bearing_y_deg=np.degrees(np.arctan2(velocity[:,1],velocity[:,2])),
        rotation_step_deg_s=np.r_[step,step[-1]],rotation_window_deg_s=rotation_angle(r[lo],r[hi])/dt,
        reference_start_index=lo,reference_end_index=hi)
