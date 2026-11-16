"""Reflection checks against independent pinhole camera geometry."""
import unittest

from mstd.mirror_controls import reflect_image, reflect_truth, pair_symmetry
from mstd.selfmotion import Camera, Motion, project


class MirrorTests(unittest.TestCase):
    def test_non_square_pixels_are_horizontal_and_round_trip(self):
        source=bytes([1,2,3,4,5,6])
        mirrored=reflect_image(source,3,2)
        self.assertEqual(mirrored,bytes([3,2,1,6,5,4]))
        self.assertEqual(reflect_image(mirrored,3,2),source)
        with self.assertRaises(ValueError):reflect_image(source,4,2)

    def test_truth_matches_reflected_camera_projection_and_flow(self):
        camera=Camera(width=20,height=16)
        motion=Motion(heading_deg=7)
        point=(1.5,2.0,12.)
        pose=motion.pose(.5); next_pose=motion.pose(.55)
        u,v,z=project(camera,point,pose); un,vn,zn=project(camera,point,next_pose)
        source=dict(frame=10,time_s=.5,camera=pose,respawned_points=0,
                    sparse_flow_to_next_frame=[dict(id=3,u_px=u,v_px=v,depth_m=z,du_px=un-u,dv_px=vn-v)])
        actual=reflect_truth(source,20)
        reflected_motion=Motion(heading_deg=-7)
        ur,vr,zr=project(camera,(-point[0],point[1],point[2]),reflected_motion.pose(.5))
        unr,vnr,znr=project(camera,(-point[0],point[1],point[2]),reflected_motion.pose(.55))
        p=actual['sparse_flow_to_next_frame'][0]
        self.assertAlmostEqual(p['u_px'],ur)
        self.assertAlmostEqual(p['v_px'],vr)
        self.assertAlmostEqual(p['du_px'],unr-ur)
        self.assertAlmostEqual(p['dv_px'],vnr-vr)
        self.assertEqual(actual['camera'],reflected_motion.pose(.5))
        self.assertEqual(source['camera']['heading_camera_deg'],7)
        self.assertEqual(actual['frame'],source['frame'])
        source['sparse_flow_to_next_frame'][0]['du_px']=None
        self.assertIsNone(reflect_truth(source,20)['sparse_flow_to_next_frame'][0]['du_px'])

    def test_direction_symmetry_separates_shared_shift_from_quantization(self):
        self.assertEqual(pair_symmetry([-5,-10],[5,10])['mean_common_bias_deg'],0)
        shifted=pair_symmetry([-2,-7],[8,13])
        self.assertEqual(shifted['mean_common_bias_deg'],3)
        self.assertEqual(shifted['mean_absolute_mirror_deviation_deg'],6)
        for left,right in (([],[]),([0],[0,1]),([float('nan')],[0])):
            with self.assertRaises(ValueError):pair_symmetry(left,right)


if __name__=='__main__':unittest.main()
