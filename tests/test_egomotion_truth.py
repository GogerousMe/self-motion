import unittest
import numpy as np
from mstd.egomotion_truth import quaternion_xyzw_rotations,camera_motion_reference


class TruthTests(unittest.TestCase):
    def test_world_heading_is_rotated_into_camera_coordinates_not_used_as_yaw(self):
        # Camera yaw is90deg in the world, but motion is along its optical z.
        q=np.tile([0,np.sin(np.pi/4),0,np.cos(np.pi/4)],(11,1))
        r=quaternion_xyzw_rotations(q);t=np.arange(11)*.1
        p=t[:,None]*r[0,:,2]
        state=camera_motion_reference(r,p,t,2)
        np.testing.assert_allclose(state['heading_x_deg'],0,atol=1e-12)
        np.testing.assert_allclose(state['speed_mps'],1,atol=1e-12)
        np.testing.assert_allclose(state['rotation_step_deg_s'],0,atol=1e-5)

    def test_reflected_translation_changes_heading_sign_and_quaternion_sign_does_not(self):
        t=np.arange(11)*.1;r=np.tile(np.eye(3),(11,1,1))
        p=np.stack((t*np.sin(np.radians(7)),t*0,t*np.cos(np.radians(7))),axis=1)
        a=camera_motion_reference(r,p,t,2)
        p[:,0]*=-1;b=camera_motion_reference(r,p,t,2)
        np.testing.assert_allclose(a['heading_x_deg'],7)
        np.testing.assert_allclose(b['heading_x_deg'],-7)
        q=np.array([[0,.1,0,.9]])
        np.testing.assert_array_equal(quaternion_xyzw_rotations(q),quaternion_xyzw_rotations(-q))


if __name__=='__main__':unittest.main()
