"""Protocol/readout tests; synthetic events are NOT a neural-dynamics test."""
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from mstd.carlsim import SIGNATURE
from mstd.model import model_config,VisionModel
from mstd.projection import load_projection
from mstd.simulation import simulate
from mstd.spiking import spike_frames,spike_readout
from mstd.v1_recording import RawWriter

PROJECTION=str(Path(__file__).resolve().parents[1]/'configs/v1_projection.json')


def events(path,w,h,values=(),depth=1):
    path.write_bytes(struct.pack('<ifiii',SIGNATURE,.2,w,h,depth)+
                     b''.join(struct.pack('<ii',*v) for v in values))


def groups(root,w,h):
    for group,n in (('v1',84),('exc',8),('inh',8)):
        events(root/f'{group}.dat',w,h,depth=n)


class SpikingMTTests(unittest.TestCase):
    def test_bin_boundaries_silent_tail_and_channel_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);groups(root,2,3)
            events(root/'exc.dat',2,3,[(0,19),(49,23),(50,19)],depth=8)
            events(root/'inh.dat',2,3,[(51,42)],depth=8)
            values=list(spike_frames(root,2,3,3,50))
            self.assertEqual(len(values),3)
            self.assertEqual(values[0][1]['exc'][3,0,1],20)
            self.assertEqual(values[0][1]['exc'][3,2,1],20)
            self.assertEqual(values[1][1]['exc'][3,0,1],20)
            self.assertEqual(values[1][1]['inh'][7,0,0],20)
            self.assertEqual(float(values[2][1]['exc'].sum()),0)

    def test_full_validation_rejects_bad_tail_ids_order_and_grid(self):
        for bad in ([(150,0)],[(0,504)],[(50,0),(49,0)]):
            with self.subTest(events=bad),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);groups(root,2,3)
                events(root/'v1.dat',2,3,bad,depth=84)
                with self.assertRaises(ValueError):list(spike_frames(root,2,3,3,50))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);groups(root,2,3)
            events(root/'exc.dat',3,2,depth=8)
            with self.assertRaises(ValueError):list(spike_frames(root,2,3,3,50))

    def test_readout_silence_direction_and_no_spatial_spreading(self):
        angles=np.arange(8)*np.pi/4
        r=np.zeros((8,3,4),np.float32)
        self.assertEqual(spike_readout(r,angles)['evidence'],0)
        r[2,1,2]=20
        output=spike_readout(r,angles)
        self.assertAlmostEqual(output['vectors'][1,1,2],20/21,places=6)
        self.assertAlmostEqual(output['activity'].sum(),20/21,places=6)
        self.assertEqual(np.count_nonzero(output['activity']),1)
        for bad in (-r,r*np.nan):
            with self.assertRaises(ValueError):spike_readout(bad,angles)

    def test_config_no_silent_fallback_and_unsupported_streaming(self):
        base=dict(mt_backend='carlsim',mt_radius=0,projection_config=PROJECTION)
        for override in ({'mt_radius':1},{'frame_ms':50.5},{'carlsim':{'seed':-1}},
                         {'carlsim':{'v1_rate_scale':float('nan')}}):
            with self.assertRaises(ValueError):model_config({**base,**override})
        with self.assertRaises(FileNotFoundError):simulate(base)
        with self.assertRaisesRegex(ValueError,'offline'):VisionModel(19,19,base)

    def test_offline_integration_uses_observed_e_spikes_for_mstd(self):
        # Mock ONLY the process boundary. Native output is a protocol fixture.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);exe=root/'bridge';exe.write_text('fixture')
            raw=RawWriter(root/'raw',2,3,.05,{'source':'test'})
            for _ in range(3):raw.append(np.zeros((3,28,3,2),np.float32))
            raw.close()
            settings=dict(mt_backend='carlsim',mt_radius=0,projection_config=PROJECTION,
                          input=dict(kind='v1_recording',path=str(root/'raw')),
                          layers=dict(v1=False),output=str(root/'out'),progress_every=0,
                          carlsim=dict(executable=str(exe)),monitors=dict(every_n_frames=2))
            def fixture(command,cwd,**kwargs):
                self.assertEqual(command[3:7],['2','3','3','50'])
                groups(cwd,2,3)
                events(cwd/'exc.dat',2,3,[(0,0),(50,1)],depth=8)
                events(cwd/'inh.dat',2,3,[(49,0)],depth=8)
                p=load_projection(PROJECTION)
                report=dict(schema='mstd-carlsim-native-v1',complete=True,width=2,height=3,
                            frames=3,frame_ms=50,seed=42,mode='cpu',
                            connections=int(6*(3*np.count_nonzero(p.weights)+8)))
                (cwd/'native_report.json').write_text(json.dumps(report))
            with patch('mstd.spiking.subprocess.run',side_effect=fixture):
                report=simulate(settings)
            self.assertTrue(report['complete'])
            self.assertEqual(report['event_counts'],dict(v1=0,exc=2,inh=1))
            self.assertIsNotNone(report['last_mstd'])
            self.assertFalse(report['last_mstd']['bearing_valid'])
            meta=json.loads((root/'out/monitors/mt_exc_rates/metadata.json').read_text())
            self.assertEqual(meta['sample_indices'],[0,2])
            self.assertEqual(meta['timestamps_s'],[.05,.15000000000000002])
            self.assertEqual(len((root/'out/estimates.jsonl').read_text().splitlines()),3)
            with self.assertRaises(FileExistsError):simulate(settings)


if __name__=='__main__':unittest.main()
