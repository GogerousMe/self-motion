import json
from math import pi
from pathlib import Path
import struct
import tempfile
import unittest

from mstd.calibration import analyze_calibration, generate_suite, inspect_stimulus, stimulus_frames
from mstd.carlsim import SIGNATURE


class CalibrationTests(unittest.TestCase):
    def test_motion_sign_is_right_down_left_up(self):
        for kind in ['grating','texture']:
            for angle,dx,dy in [(0,1,0),(90,0,1),(180,-1,0),(270,0,-1)]:
                frames = list(stimulus_frames(16,16,2,angle,1,kind=kind,seed=3))
                for y in range(1,15):
                    for x in range(1,15):
                        self.assertEqual(frames[1][y*16+x],frames[0][(y-dy)*16+x-dx])

    def test_generation_layout_determinism_and_pending_is_not_passed(self):
        config = Path(__file__).resolve().parents[1]/'configs/v1_projection.json'
        with tempfile.TemporaryDirectory() as tmp:
            for name in ['a','b']:
                generate_suite(Path(tmp)/name,width=8,height=6,frames=3,warmup_frames=1)
            a,b = Path(tmp)/'a',Path(tmp)/'b'
            file = a/'grating_d000.dat'
            self.assertEqual(file.read_bytes(),(b/file.name).read_bytes())
            self.assertEqual(inspect_stimulus(file),dict(width=8,height=6,frames=3,channels=1,type=4))
            report = analyze_calibration(a/'manifest.json',config,border=0)
            self.assertEqual(report['pending_runs'],8)
            self.assertFalse(report['calibration_passed'])
            with self.assertRaises(FileExistsError):
                generate_suite(a)

    def test_known_filter_responses_calibrate_and_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)/'suite'
            generate_suite(directory,width=4,height=4,frames=2,warmup_frames=1)
            manifest = directory/'manifest.json'
            meta = json.loads(manifest.read_text())
            weights = [[float(i==j) for j in range(8)] for i in range(28)]
            config = Path(tmp)/'projection.json'
            config.write_text(json.dumps({'schema':'v1-filter-projection-v1','gain':1,
                'scale_weights':[1,1,1],'directions_rad':[-i*pi/4 for i in range(8)],
                'projections':{'test':{'weights':weights}}}))
            for run in meta['runs']:
                channel = (-run['expected_direction_deg']//45)%8
                path = directory/(run['name']+'_spikes.dat')
                # Place activity in a known filter/pixel, after warmup.
                path.write_bytes(struct.pack('<ifiii',SIGNATURE,.2,4,4,84)
                                 +struct.pack('<ii',50,channel*16+5))
                run['spikes'] = path.name
            manifest.write_text(json.dumps(meta))
            report = analyze_calibration(manifest,config,'test',border=0)
            self.assertTrue(report['calibration_passed'])
            self.assertTrue(all(r['winner_error_deg']==0 for r in report['runs']))
            self.assertTrue(all(len(r['filter_mean_hz'])==84 for r in report['runs']))
            stimulus = directory/meta['runs'][0]['stimulus']
            data = bytearray(stimulus.read_bytes())
            data[-1] ^= 1
            stimulus.write_bytes(data)
            with self.assertRaisesRegex(ValueError,'checksum'):
                analyze_calibration(manifest,config,'test',border=0)

    def test_invalid_geometry_timing_and_file_size(self):
        config = Path(__file__).resolve().parents[1]/'configs/v1_projection.json'
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)/'suite'
            with self.assertRaises(ValueError):
                generate_suite(directory,frames=10,warmup_frames=20)
            self.assertFalse(directory.exists())
            generate_suite(directory,width=8,height=8,frames=2,warmup_frames=1)
            with self.assertRaises(ValueError):
                analyze_calibration(directory/'manifest.json',config,border=4)
            path = directory/'grating_d000.dat'
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                inspect_stimulus(path)


if __name__ == '__main__':
    unittest.main()
