"""Pixel, order, geometry and failure checks for real image-folder conversion."""
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from mstd.calibration import file_sha256, inspect_stimulus
from mstd.image_sequence import convert_images

HAS_PIL = importlib.util.find_spec('PIL') is not None
if HAS_PIL:
    from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(HAS_PIL, 'image conversion tests require optional Pillow')
class ImageSequenceTests(unittest.TestCase):
    def test_numeric_order_rgb_layout_metadata_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); folder = root/'images'; folder.mkdir()
            colors = {'in10.png': (111, 122, 133), 'in2.png': (22, 33, 44), 'in1.png': (1, 2, 3)}
            for name, color in colors.items():
                Image.new('RGB', (19, 21), color).save(folder/name)
            before = {name: file_sha256(folder/name) for name in colors}
            (folder/'notes.txt').write_text('ignored', encoding='utf-8')
            output = root/'movie.dat'
            report = convert_images(folder, output, mode='rgb', frame_ms=40, progress_every=0)
            self.assertEqual(inspect_stimulus(output), dict(width=19, height=21, channels=3, frames=3, type=5))
            payload = output.read_bytes()
            self.assertEqual(struct.unpack('<ifiBiii', payload[:25]), (293390619, 1., 5, 3, 19, 21, 3))
            self.assertEqual(payload[25:], b''.join(bytes(colors[name])*19*21 for name in ('in1.png', 'in2.png', 'in10.png')))
            self.assertEqual([v['source'] for v in report['frame_order']], ['in1.png', 'in2.png', 'in10.png'])
            self.assertEqual([v['timestamp_s'] for v in report['frame_order']], [0., .04, .08])
            self.assertEqual(report['fps'], 25)
            self.assertEqual(report['stimulus_sha256'], file_sha256(output))
            with self.assertRaises(FileExistsError):
                convert_images(folder, output, progress_every=0)
            self.assertEqual(output.read_bytes(), payload)
            self.assertEqual({name: file_sha256(folder/name) for name in colors}, before)

    def test_center_padding_grayscale_alpha_and_explicit_resize(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); folder = root/'images'; folder.mkdir()
            Image.new('L', (19, 21), 120).save(folder/'frame1.png')
            output = root/'padded.dat'
            report = convert_images(folder, output, geometry='pad', size=(27, 21), progress_every=0)
            payload = output.read_bytes()[25:]
            self.assertEqual(payload, (bytes(4)+bytes([120])*19+bytes(4))*21)
            self.assertEqual(report['frame_order'][0]['pad_left'], 4)
            self.assertEqual(report['frame_order'][0]['pad_top'], 0)
            resized = root/'resized.dat'
            convert_images(folder, resized, geometry='resize', size=(23, 25), progress_every=0)
            self.assertEqual(resized.read_bytes()[25:], bytes([120])*23*25)
            Image.new('RGBA', (19, 21), (255, 0, 0, 0)).save(folder/'frame2.png')
            alpha = root/'alpha.dat'
            convert_images(folder, alpha, mode='rgb', progress_every=0)
            self.assertEqual(alpha.read_bytes()[25+19*21*3:], bytes(19*21*3))

    def test_invalid_geometry_mixed_sizes_and_corrupt_images_leave_no_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); folder = root/'images'; folder.mkdir()
            Image.new('L', (19, 21), 10).save(folder/'frame1.png')
            Image.new('L', (21, 19), 20).save(folder/'frame2.png')
            for options in ({}, {'geometry': 'pad', 'size': (19, 19)}, {'size': (27, 21)},
                            {'frame_ms': 0}, {'geometry': 'resize'}, {'max_frames': 0}):
                with self.subTest(options=options):
                    output = root/'bad.dat'
                    with self.assertRaises(ValueError):
                        convert_images(folder, output, progress_every=0, **options)
                    self.assertFalse(output.exists())
                    self.assertFalse(output.with_suffix('.dat.json').exists())
            # A PNG header can pass preflight while its pixel stream is corrupt.
            path = folder/'frame2.png'
            path.write_bytes(path.read_bytes()[:41])
            output = root/'corrupt.dat'
            with self.assertRaises((OSError, ValueError)):
                convert_images(folder, output, geometry='resize', size=(21, 21), progress_every=0)
            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix('.dat.json').exists())

    def test_cli_runs_directly_with_pattern_and_frame_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); folder = root/'images'; folder.mkdir()
            for index in range(1, 4):
                Image.new('L', (19, 21), index).save(folder/f'in{index}.png')
            output = root/'selected.dat'
            command = [sys.executable, str(ROOT/'examples/images_to_dat.py'), '--input', str(folder),
                       '--output', str(output), '--pattern', '*.png', '--max-frames', '2', '--fps', '25']
            run = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout+run.stderr)
            meta = json.loads(output.with_suffix('.dat.json').read_text(encoding='utf-8'))
            self.assertEqual((meta['frames'], meta['available_images'], meta['frame_ms']), (2, 3, 40))
            self.assertEqual(output.read_bytes()[25:], bytes([1])*19*21+bytes([2])*19*21)


if __name__ == '__main__':
    unittest.main()
