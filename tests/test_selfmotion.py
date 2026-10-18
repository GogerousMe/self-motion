import json
from math import hypot
from pathlib import Path
import tempfile
import unittest

from mstd.calibration import inspect_stimulus
from mstd.selfmotion import Camera, Motion, generate_selfmotion, project, sequence


class SelfMotionTests(unittest.TestCase):
    def test_pinhole_translation_expansion_and_contraction(self):
        camera = Camera()
        point = (2,camera.height_m,10)
        u0 = project(camera,point,Motion().pose(0))[0]
        uf = project(camera,point,Motion().pose(.05))[0]
        ub = project(camera,point,Motion('backward').pose(.05))[0]
        self.assertGreater(uf,u0)
        self.assertLess(ub,u0)
        self.assertAlmostEqual(u0-camera.cx,camera.fx*2/10)
        self.assertEqual(Motion('backward').pose(0)['heading_camera_deg'],180)

    def test_yaw_sign_and_curved_path_geometry(self):
        camera = Camera()
        point = (0,camera.height_m,10)
        right = Motion('yaw',yaw_rate_deg_s=10).pose(.1)
        self.assertLess(project(camera,point,right)[0],camera.cx)
        self.assertEqual(right['x_m'],0)
        self.assertIsNone(right['heading_camera_deg'])
        for sign in [-1,1]:
            motion = Motion('curve',radius_m=20,turn_sign=sign,gaze_offset_deg=15)
            pose = motion.pose(2)
            self.assertAlmostEqual(hypot(pose['x_m']-sign*20,pose['z_m']),20)
            self.assertAlmostEqual(hypot(pose['vx_mps'],pose['vz_mps']),3)
            self.assertAlmostEqual(pose['heading_camera_deg'],-15)
            self.assertEqual(pose['signed_curvature_per_m'],sign/20)

    def test_static_frames_ground_horizon_and_exact_sparse_displacement(self):
        camera = Camera(width=16,height=16)
        static = list(sequence(camera,Motion('static'),'ground',3,50,10,2))
        self.assertEqual(static[0][0],static[1][0])
        self.assertTrue(all(p['v_px']>camera.cy for p in static[0][1]['sparse_flow_to_next_frame']))
        self.assertTrue(all(p['du_px']==0 and p['dv_px']==0 for p in static[0][1]['sparse_flow_to_next_frame']))
        moving = list(sequence(camera,Motion(),'cloud',3,50,10,2))
        first = moving[0][1]['sparse_flow_to_next_frame']
        second = {p['id']:p for p in moving[1][1]['sparse_flow_to_next_frame']}
        for point in first:
            if point['du_px'] is not None:
                later = second[point['id']]
                self.assertAlmostEqual(point['u_px']+point['du_px'],later['u_px'])
                self.assertAlmostEqual(point['v_px']+point['dv_px'],later['v_px'])

    def test_generated_bundle_records_geometry_truth_and_exact_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'suite'
            result = generate_selfmotion(out,frames=2,points=5)
            self.assertEqual(result['runs'],14)
            meta = json.loads((out/'manifest.json').read_text())
            self.assertEqual(meta['intrinsics_px']['fx_px'],Camera().fx)
            for run in meta['runs']:
                self.assertEqual(inspect_stimulus(out/run['stimulus'])['frames'],2)
                truth = [json.loads(line) for line in (out/run['truth']).read_text().splitlines()]
                self.assertEqual(len(truth),2)
                self.assertEqual(truth[0]['time_s'],0)
                self.assertTrue(all(p['du_px'] is None for p in truth[-1]['sparse_flow_to_next_frame']))
                self.assertIsNone(run['spikes'])
            with self.assertRaises(FileExistsError):
                generate_selfmotion(out,frames=2,points=5)

    def test_invalid_geometry_and_motion(self):
        for kwargs in [dict(near_m=0),dict(fov_x_deg=180),dict(height_m=-1)]:
            with self.assertRaises(ValueError):
                Camera(**kwargs)
        with self.assertRaises(ValueError):
            Motion('curve',radius_m=0)
        with self.assertRaises(ValueError):
            list(sequence(Camera(),Motion(),'ground',frames=0))


if __name__ == '__main__':
    unittest.main()
