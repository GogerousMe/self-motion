"""Passive layer monitors: sampling only affects saved data, never model updates."""
import csv
import json
from pathlib import Path

import numpy as np

from .v1_recording import RawWriter,dump_json


class ArrayMonitor:
    def __init__(self,path,shape,layout,units,extra=None):
        self.path=Path(path)
        self.path.mkdir(parents=True,exist_ok=False)
        self.shape=tuple(shape)
        self.layout,self.units=layout,units
        self.extra=extra or {}
        self.stream=(self.path/'values.f32').open('xb')
        self.indices,self.times=[],[]

    def append(self,value,index,timestamp_s):
        value=np.asarray(value)
        if value.shape!=self.shape or not np.isfinite(value).all():
            raise ValueError('invalid monitor array')
        self.stream.write(value.astype('<f4',copy=False).tobytes())
        self.indices.append(index);self.times.append(timestamp_s)

    def close(self,success):
        self.stream.close()
        if success:
            dump_json(self.path/'metadata.json',{
                'schema':'mstd-layer-monitor-v1','complete':True,'file':'values.f32',
                'dtype':'float32-le','layout':'T'+self.layout,'units':self.units,
                'shape':[len(self.indices),*self.shape],
                'sample_indices':self.indices,'timestamps_s':self.times,**self.extra})


class ModelMonitors:
    def __init__(self,output,model,provenance):
        self.output,self.model=Path(output),model
        self.config=model.config['monitors']
        self.stride=self.config['every_n_frames']
        self.arrays={}
        self.raw=None
        self.raw_indices=[]
        self.raw_times=[]
        self.summary_stream,self.estimates_stream=None,None
        self.spike_streams={}
        self.spike_records={}
        self.spiking=model.config['mt_backend']=='python_spiking'
        self.raw_provenance={**provenance,'monitor_every_n_frames':self.stride}
        root=self.output/'monitors'
        directions=list(model.projection.directions_rad) if model.projection else None
        if self.config['v1']:
            self.raw=RawWriter(root/'v1',model.width,model.height,model.interval*self.stride,self.raw_provenance)
        for name,key,units in (('mt_projection','direction_rates','Hz'),('mt_pooling','pooled_rates','Hz'),
                               ('mt_normalization','activity','dimensionless')):
            if self.config[name] and model.mt:
                target=({'mt_projection':'mt_exc_rates','mt_pooling':'mt_filtered_rates','mt_normalization':'mt_readout_activity'}[name]
                        if self.spiking else name)
                extra={'directions_rad':directions,'coordinates':'x-right_y-down'}
                if self.spiking:extra.update(timestamp_reference='frame_end',semantic={
                    'direction_rates':'observed MT excitatory spike count / frame duration',
                    'pooled_rates':'causal exponential rate trace; no spatial pooling',
                    'activity':'rate readout adapter; not neural divisive normalization'}[key])
                monitor=ArrayMonitor(root/target,(8,model.height,model.width),'DYX',units,extra)
                self.arrays[target]=(monitor,key)
        if self.spiking:
            if self.config['mt_projection']:
                self.arrays['mt_inh_rates']=(ArrayMonitor(root/'mt_inh_rates',(8,model.height,model.width),'DYX','Hz',
                     {'timestamp_reference':'frame_end','directions_rad':directions}),'inh_rates')
            if self.config['neural_state']:
                for key,units in (('exc_voltage','mV'),('inh_voltage','mV'),('exc_recovery','model units'),
                                  ('inh_recovery','model units'),('exc_conductance','effective conductance'),('inh_conductance','effective conductance')):
                    name='mt_'+key
                    self.arrays[name]=(ArrayMonitor(root/name,(8,model.height,model.width),'DYX',units,
                                                  {'timestamp_reference':'frame_end'}),key)
            if self.config['spikes']:
                event_root=root/'spikes';event_root.mkdir(parents=True,exist_ok=False)
                for group in ('v1','exc','inh'):
                    self.spike_streams[group]=(event_root/f'{group}.events').open('xb')
                    self.spike_records[group]=0
        if self.config['mstd'] and model.mstd:
            extra={'centers_px':[list(c) for c in model.centers],
                   'bearing_calibrated':model.intrinsics is not None}
            self.arrays['mstd_activity']=(ArrayMonitor(root/'mstd_activity',(len(model.centers),),'K','dimensionless',extra),'activity')
            self.arrays['mstd_scores']=(ArrayMonitor(root/'mstd_scores',(len(model.centers),),'K','dimensionless',extra),'template_scores')
            self.estimates_stream=(self.output/'estimates.jsonl').open('x',encoding='utf-8')
        if self.config['summary']:
            self.summary_stream=(self.output/'frames.csv').open('x',newline='',encoding='utf-8')
            self.summary_writer=csv.DictWriter(self.summary_stream,fieldnames=[
                'frame','timestamp_s','available_timestamp_s','compute_s','v1_mean_hz','mt_evidence',
                'mstd_winner','mstd_margin','bearing_valid','alpha_x_deg','alpha_y_deg'])
            self.summary_writer.writeheader()

    def record(self,state,index,compute_s):
        if index%self.stride:return
        op=self.model.ops
        time=state['timestamp_s']
        if self.raw:
            self.raw.append(op.host(state['v1']))
            self.raw_indices.append(index);self.raw_times.append(time)
        for name,(monitor,key) in self.arrays.items():
            if name.startswith('mt_'):
                value=op.host(state['mt'][key])
            else:
                value=np.asarray(state['mstd'][key])
            monitor.append(value,index,state['available_timestamp_s'] if self.spiking else time)
        mst=state['mstd']
        if self.estimates_stream:
            self.estimates_stream.write(json.dumps({**mst,'frame':index,'input_timestamp_s':time,
                'estimate_timestamp_s':state['available_timestamp_s']},allow_nan=False)+'\n')
        if self.summary_stream:
            mean=float(state['v1'].mean())
            self.summary_writer.writerow({'frame':index,'timestamp_s':time,
                'available_timestamp_s':state['available_timestamp_s'],'compute_s':compute_s,
                'v1_mean_hz':mean,'mt_evidence':state['mt']['evidence'] if state['mt'] else None,
                'mstd_winner':mst['winner_index'] if mst else None,'mstd_margin':mst['margin'] if mst else None,
                'bearing_valid':mst['bearing_valid'] if mst else False,
                'alpha_x_deg':mst['alpha_x_deg'] if mst else None,'alpha_y_deg':mst['alpha_y_deg'] if mst else None})

    def record_spikes(self,tick,timestamp_s,events):
        """Passive sparse count events. Not CARLsim DAT; one record per nonzero bin/neuron."""
        dtype=np.dtype([('tick','<u8'),('neuron','<u4'),('count','<u4')])
        for group,stream in self.spike_streams.items():
            counts=self.model.ops.host(events[group]).reshape(-1)
            ids=np.flatnonzero(counts)
            if len(counts)>2**32 or (counts[ids]>np.iinfo(np.uint32).max).any():raise ValueError('spike event exceeds uint32 range')
            rows=np.empty(len(ids),dtype=dtype)
            rows['tick']=tick;rows['neuron']=ids;rows['count']=counts[ids]
            stream.write(rows.tobytes());self.spike_records[group]+=len(ids)

    def close(self,success):
        try:
            if self.raw:
                self.raw.close(success)
                if success and self.raw_indices:
                    path=self.raw.path/'metadata.json'
                    meta=json.loads(path.read_text(encoding='utf-8'))
                    meta.update({'sample_indices':self.raw_indices,'timestamps_s':self.raw_times,
                                 'monitor_every_n_frames':self.stride})
                    dump_json(path,meta)
            for monitor,_ in self.arrays.values():monitor.close(success)
        finally:
            for stream in self.spike_streams.values():stream.close()
            if success and self.spike_streams:
                dump_json(self.output/'monitors/spikes/metadata.json',dict(
                    schema='mstd-python-spike-count-events-v1',complete=True,record_bytes=16,
                    fields=['tick:uint64-le','neuron:uint32-le','count:uint32-le'],
                    dt_ms=self.model.config['spiking']['dt_ms'],ticks=self.model.ticks,
                    timestamp_reference='(tick+1)*dt_ms, tick end; sequence starts at 0',
                    records=self.spike_records,files={g:g+'.events' for g in self.spike_streams},
                    shapes={'v1':[3,28,self.model.height,self.model.width],
                            'exc':[8,self.model.height,self.model.width],'inh':[8,self.model.height,self.model.width]},
                    note='all ticks recorded regardless of frame monitor stride; multi-events retain count'))
            if self.summary_stream:self.summary_stream.close()
            if self.estimates_stream:self.estimates_stream.close()

    def description(self):
        active={'v1':self.raw is not None,'summary':self.summary_stream is not None,
                'mstd':'mstd_activity' in self.arrays}
        active.update({name:({'mt_projection':'mt_exc_rates','mt_pooling':'mt_filtered_rates','mt_normalization':'mt_readout_activity'}[name]
                            if self.spiking else name) in self.arrays for name in ('mt_projection','mt_pooling','mt_normalization')})
        active.update(spikes=bool(self.spike_streams),neural_state='mt_exc_voltage' in self.arrays)
        return {'requested':self.config,'active':active,'saved_arrays':list(self.arrays),
            'every_n_frames':self.stride,
            'note':'unrecorded frames are still processed; disabled layer monitors are inactive'}
