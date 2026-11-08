"""Verify real model-monitor/video pairing, stage values and encoded frame count."""
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from mstd.calibration import file_sha256, stimulus_frames, write_stimulus
from mstd.simulation import simulate

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT/'examples'
sys.path.insert(0, str(EXAMPLES))
try:
    SPEC = importlib.util.spec_from_file_location('render_model_mt', EXAMPLES/'render_model_mt.py')
    VIDEO = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(VIDEO)
finally:
    sys.path.pop(0)


class ModelVideoTests(unittest.TestCase):
    def fixture(self, root, stride=1, count=3):
        stimulus = root/'input.dat'
        write_stimulus(stimulus, 19, 21, stimulus_frames(19, 21, 4, 0, 1.5))
        output = root/'model'
        simulate({'input': {'path': str(stimulus)}, 'output': str(output),
                  'projection_config': str(ROOT/'configs/v1_projection.json'),
                  'layers': {'mstd': False}, 'monitors': {'v1': False, 'every_n_frames': stride},
                  'max_frames': count, 'progress_every': 0})
        return stimulus, output

    def test_recording_pairing_and_reject_missing_frames_sparse_wrong_input_and_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stimulus, output = self.fixture(root)
            data = VIDEO.load_recording(output, stimulus)
            self.assertEqual(data['frames'], 3)
            self.assertEqual(data['info']['frames'], 4)
            with self.assertRaisesRegex(ValueError, 'only 3 model frames'):
                VIDEO.load_recording(output, stimulus, max_frames=4)
            different = root/'different.dat'
            raw = bytearray(stimulus.read_bytes()); raw[-1] ^= 1
            different.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                VIDEO.load_recording(output, different)
            binary = output/'monitors/mt_pooling/values.f32'
            binary.write_bytes(binary.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, 'size mismatch'):
                VIDEO.load_recording(output, stimulus)
        with tempfile.TemporaryDirectory() as tmp:
            stimulus, output = self.fixture(Path(tmp), stride=2)
            with self.assertRaisesRegex(ValueError, 'every_n_frames=1'):
                VIDEO.load_recording(output, stimulus)

    def test_rgb_renderer_preserves_axes_and_fixed_direction_color_scale(self):
        renderer = VIDEO.Renderer(19, 21, 300, 1, 'jet')
        pixels = np.zeros((21, 19, 3), np.uint8)
        pixels[..., 0] = 255
        maps = np.zeros((3, 8, 21, 19), np.float32)
        maps[0, 2, :, :] = 150
        maps[2, 2, :, :] = .5
        canvas = renderer.render(pixels, maps, 0, 4, .05)
        self.assertEqual(canvas.shape, (1230, 1440, 3))
        np.testing.assert_array_equal(canvas[100, 100], [255, 0, 0])
        # Same relative amplitude must have the same color in Hz/activity rows.
        np.testing.assert_array_equal(canvas[400, 20+2*177+80], renderer.colors(150, 300))
        np.testing.assert_array_equal(canvas[1010, 20+2*177+80], renderer.colors(.5, 1))
        np.testing.assert_array_equal(canvas[400, 100], renderer.colors(0, 300))

    def test_optional_manifest_requires_actual_stimulus_and_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stimulus, output = self.fixture(root)
            segments = root/'segments.csv'
            segments.write_text('segment_index,first_frame,last_frame,start_ms,end_ms,is_blank,direction_slot_deg,speed_px_per_frame\n'
                                '1,1,4,0,200,0,0,1.5\n', encoding='utf-8')
            manifest = root/'manifest.json'
            meta = dict(schema='matlab-sequential-binary-gratings-v2',
                        coordinates='x-right_y-down; clockwise from right', width=19, height=21, frame_ms=50,
                        runs=dict(frames=4, duration_ms=200, stimulus_sha256=file_sha256(stimulus), segments=segments.name))
            manifest.write_text(json.dumps(meta), encoding='utf-8')
            data = VIDEO.load_recording(output, stimulus, manifest)
            self.assertEqual(data['segments'][0]['direction_slot_deg'], 0)
            meta['frame_ms'] = 100
            manifest.write_text(json.dumps(meta), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'timing mismatch'):
                VIDEO.load_recording(output, stimulus, manifest)

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'video integration requires ffmpeg/ffprobe')
    def test_cli_encodes_all_saved_frames_and_statistics_match_monitors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stimulus, model = self.fixture(root)
            original = file_sha256(model/'monitors/mt_normalization/values.f32')
            out = root/'video'
            command = [sys.executable, str(EXAMPLES/'compare_mt_stages.py'), '--recording', str(model),
                       '--stimulus', str(stimulus), '--output', str(out), '--progress-every', '0']
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                                    '-show_entries', 'stream=nb_frames,r_frame_rate,width,height', '-of', 'json',
                                    str(out/'mt_stages.mp4')], capture_output=True, text=True, check=True)
            video = json.loads(probe.stdout)['streams'][0]
            self.assertEqual((video['width'], video['height'], video['nb_frames'], video['r_frame_rate']),
                             (1440, 1230, '3', '20/1'))
            config = json.loads((out/'video_config.json').read_text())
            self.assertFalse(config['complete_stimulus'])
            self.assertEqual(config['frames'], 3)
            with (out/'stage_statistics.csv').open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 9)
            data = VIDEO.load_recording(model, stimulus)
            for s, (binary, shape, _) in enumerate(data['definitions']):
                maps = np.fromfile(binary, dtype='<f4').reshape(shape)
                for i in range(3):
                    expected = maps[i][data['order']].mean(axis=(-2, -1), dtype=np.float64)
                    row = rows[i*3+s]
                    self.assertEqual(row['input_index'], str(i))
                    np.testing.assert_allclose([float(row[f'mean_{d}_deg']) for d in range(0, 360, 45)],
                                               expected, rtol=0, atol=0)
            self.assertEqual(file_sha256(model/'monitors/mt_normalization/values.f32'), original)
            # Never silently turn a short preview into a requested full recording.
            result = subprocess.run(command[:-2]+['--max-frames', '4'], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('only 3 model frames', result.stderr)


if __name__ == '__main__':
    unittest.main()
