import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from mstd.calibration import generate_suite, write_stimulus, file_sha256, inspect_stimulus

EXAMPLES=Path(__file__).resolve().parents[1]/'examples'
SPEC=importlib.util.spec_from_file_location('prepare_validation_sequences',EXAMPLES/'prepare_validation_sequences.py')
PREP=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREP)


class ValidationSequenceTests(unittest.TestCase):
    def test_continuous_direction_change_carries_spatial_position(self):
        frames=list(PREP.continuous_texture(8,8,2,1.,0))
        self.assertEqual([d for d,_ in frames], [d for d in range(0,360,45) for _ in range(2)])
        # First 90-degree sample moves down from last45 sample, without phase reset.
        before,after=frames[3][1],frames[4][1]
        for y in range(8):
            for x in range(8):
                self.assertEqual(after[y*8+x],before[((y-1)%8)*8+x])

    def test_payload_concatenation_static_controls_and_local_truth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            texture=root/'texture'; motion=root/'motion'; motion.mkdir()
            generate_suite(texture,width=8,height=8,frames=3,warmup_frames=1)
            # Actual texture generator, not grating.
            texture2=root/'texture2'
            generate_suite(texture2,width=8,height=8,frames=3,warmup_frames=1,kind='texture')
            runs=[]
            for i,name in enumerate(PREP.BASELINE+['cloud_backward']):
                path=motion/(name+'.dat')
                write_stimulus(path,8,8,[bytes([i*20+t]*64) for t in range(3)])
                truth=motion/(name+'.truth.jsonl')
                truth.write_text(''.join(json.dumps({'frame':t,'time_s':t*.05})+'\n' for t in range(3)))
                runs.append(dict(name=name,stimulus=path.name,stimulus_sha256=file_sha256(path),
                    truth=truth.name,scene='cloud',motion=dict(mode='static' if i==0 else 'straight',heading_deg=0)))
            (motion/'intrinsics.json').write_text('{"fx_px":4,"fy_px":4,"cx_px":3.5,"cy_px":3.5}')
            (motion/'manifest.json').write_text(json.dumps(dict(schema='selfmotion-stimulus-v1',
                width=8,height=8,frames=3,frame_ms=50,duration_ms=150,runs=runs,
                camera={},intrinsics_px=dict(fx_px=4,fy_px=4,cx_px=3.5,cy_px=3.5))))
            out=root/'out'
            PREP.prepare(texture2,motion,out)
            tex_meta=json.loads((out/'texture/manifest.json').read_text())
            self.assertEqual(inspect_stimulus(out/'texture/all_stimuli.dat')['frames'],568)
            images=list(PREP.read_images(out/'texture/all_stimuli.dat'))
            original=list(PREP.read_images(texture2/'texture_d000.dat'))
            self.assertEqual(images[:80],[original[0]]*80)
            self.assertEqual(images[80:83],original)
            self.assertEqual(images[83:123],[original[-1]]*40)
            self.assertTrue(all(any(f) for f in images))  # no artificial black intervals
            motion_images=list(PREP.read_images(out/'selfmotion/all_stimuli.dat'))
            expected=[f for name in PREP.BASELINE for f in PREP.read_images(motion/(name+'.dat'))]
            self.assertEqual(motion_images,expected)
            truth=[json.loads(line) for line in (out/'selfmotion/truth.jsonl').read_text().splitlines()]
            self.assertEqual(len(truth),15)
            self.assertEqual(sum(r['scene_cut'] for r in truth),5)
            self.assertEqual(truth[-1]['combined_frame'],15)
            self.assertTrue(all(r['condition']!='cloud_backward' for r in truth))
            self.assertEqual(tex_meta['runs']['stimulus_sha256'],file_sha256(out/'texture/all_stimuli.dat'))
            with self.assertRaises(FileExistsError):
                PREP.prepare(texture2,motion,out)
