"""Compile/capture-format checks using a mock library, never a CUDA equivalence claim."""
from pathlib import Path
import json
import shutil
import struct
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


@unittest.skipUnless(shutil.which('g++'),'optional exporter compile test requires g++')
class ExporterTests(unittest.TestCase):
    def test_capture_layout_failures_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'motion_energy.h').write_text('''#pragma once
class MotionEnergy {
    int n;
public:
    MotionEnergy(int w,int h,int c):n(w*h*84){}
    double getScaleV1Complex(){return 25;}
    void calcV1complex(unsigned char* frame,float* out,double speed,bool gpu){
        for(int i=0;i<n;++i)out[i]=float(i%84)+frame[0];
    }
};
''',encoding='utf-8')
            exe=root/('capture.exe' if __import__('os').name=='nt' else 'capture')
            build=subprocess.run(['g++','-std=c++11','-O1',str(ROOT/'examples/export_v1_continuous.cpp'),
                                  '-I'+str(root),'-o',str(exe)],capture_output=True,text=True)
            self.assertEqual(build.returncode,0,build.stderr)
            stimulus=root/'sample.dat'
            stimulus.write_bytes(struct.pack('<ifiBiii',293390619,1.,4,1,19,19,2)+bytes([7])*19*19*2)
            output=root/'result'
            run=subprocess.run([str(exe),str(stimulus),str(output),'50','1'],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            meta=json.loads((output/'capture.json').read_text())
            self.assertEqual(meta,dict(width=19,height=19,channels=1,frames=1,frame_interval_s=.05))
            payload=(output/'rates.f32').read_bytes()
            self.assertEqual(len(payload),84*19*19*4)
            self.assertEqual(struct.unpack('<f',payload[:4])[0],7)
            overwrite=subprocess.run([str(exe),str(stimulus),str(output),'50'],capture_output=True)
            self.assertNotEqual(overwrite.returncode,0)
            self.assertEqual((output/'rates.f32').read_bytes(),payload)
            # Content-category flags do not select a different pixel encoding.
            for kind in range(7):
                with self.subTest(kind=kind):
                    stimulus.write_bytes(struct.pack('<ifiBiii',293390619,1.,kind,3,19,21,2)
                                         +bytes([7])*19*21*3*2)
                    typed_output=root/('type_'+str(kind))
                    run=subprocess.run([str(exe),str(stimulus),str(typed_output),'50'],
                                       capture_output=True,text=True)
                    self.assertEqual(run.returncode,0,run.stderr)
                    self.assertEqual((typed_output/'rates.f32').stat().st_size,84*19*21*2*4)
                    self.assertEqual(json.loads((typed_output/'capture.json').read_text())['channels'],3)
            for kind in (-1,7):
                stimulus.write_bytes(struct.pack('<ifiBiii',293390619,1.,kind,1,19,19,1)+bytes(19*19))
                invalid=root/('invalid_'+str(kind))
                run=subprocess.run([str(exe),str(stimulus),str(invalid),'50'],capture_output=True)
                self.assertNotEqual(run.returncode,0)
                self.assertFalse(invalid.exists())
            # A missing parent must fail without creating an output tree.
            stimulus.write_bytes(struct.pack('<ifiBiii',293390619,1.,6,1,19,19,1)+bytes(19*19))
            run=subprocess.run([str(exe),str(stimulus),str(root/'missing/result'),'50'],capture_output=True)
            self.assertNotEqual(run.returncode,0)
            self.assertFalse((root/'missing').exists())
            stimulus.write_bytes(stimulus.read_bytes()[:-1])
            broken=subprocess.run([str(exe),str(stimulus),str(root/'bad'),'50'],capture_output=True)
            self.assertNotEqual(broken.returncode,0)
            self.assertFalse((root/'bad').exists())


if __name__=='__main__':unittest.main()
