"""Offline V1 -> MT runner and paired continuous-response validation."""
import csv
import json
from math import atan2,degrees,hypot,isfinite
from pathlib import Path
from time import perf_counter

import numpy as np

from .calibration import file_sha256,inspect_stimulus
from .data import V1Spec
from .mstd import horizontal_centers
from .projection import load_projection
from .tensor_pipeline import DirectionMT,TensorMSTd
from .v1 import ArrayOps,MotionEnergyV1
from .v1_recording import RawWriter,dump_json,iter_raw,iter_stimulus,raw_info


def _direction_summary(values,angles):
    total = float(sum(values))
    vx = sum(float(v)*np.cos(a) for v,a in zip(values,angles))
    vy = sum(float(v)*np.sin(a) for v,a in zip(values,angles))
    resultant = hypot(vx,vy)/total if total else 0.0
    active = total>1e-12 and resultant>=0.1
    return {'winner_deg':float(degrees(angles[int(np.argmax(values))])%360) if active else None,
            'population_deg':degrees(atan2(vy,vx))%360 if active else None,
            'resultant':resultant}


def run_frontend(stimulus,output,projection_path,backend='numpy',device='cpu',frame_ms=50,
                 max_frames=None,save_raw=False,save_mt=False,projection_name='motionProj1',
                 radius=1,border=4,warmup_frames=20,expected_direction_deg=None,intrinsics=None,
                 progress_every=20):
    if not isfinite(frame_ms) or frame_ms<=0:
        raise ValueError('positive frame_ms required')
    info = inspect_stimulus(stimulus)
    w,h = info['width'],info['height']
    if type(border) is not int or not 0<=border<min(w,h)/2:
        raise ValueError('invalid evaluation ROI border')
    if type(warmup_frames) is not int or warmup_frames<0:
        raise ValueError('nonnegative warmup_frames required')
    if expected_direction_deg is not None and not isfinite(expected_direction_deg):
        raise ValueError('finite expected direction required')
    if max_frames is not None and (type(max_frames) is not int or max_frames<=0):
        raise ValueError('positive integer max_frames required')
    if type(progress_every) is not int or progress_every<0:
        raise ValueError('nonnegative progress interval required')
    count = min(info['frames'],max_frames) if max_frames else info['frames']
    projection = load_projection(projection_path,projection_name)
    model = MotionEnergyV1(w,h,frame_ms/1000,backend,device)
    op = model.ops
    mt_model = DirectionMT(op,projection,w,h,radius)
    spec,mst = None,None
    if intrinsics is not None:
        spec = V1Spec(w,h,projection.directions_rad,**intrinsics)
        mst = TensorMSTd(op,spec,horizontal_centers(spec,range(-20,21,5)))
    directory = Path(output)
    directory.mkdir(parents=True,exist_ok=False)
    provenance = model.provenance()
    provenance.update({'stimulus_sha256':file_sha256(stimulus),'stimulus':str(stimulus),
                       'complete_stimulus':count==info['frames']})
    raw = RawWriter(directory/'raw_v1',w,h,frame_ms/1000,provenance) if save_raw else None
    direction_stream = (directory/'direction_rates.f32').open('xb')
    activity_stream = (directory/'mt_activity.f32').open('xb') if save_mt else None
    metrics_stream = (directory/'frames.csv').open('x',encoding='utf-8',newline='')
    estimate_stream = (directory/'estimates.jsonl').open('x',encoding='utf-8') if mst else None
    fields = ['frame','timestamp_s','available_timestamp_s','v1_s','mt_mstd_s','mt_evidence',
              'mt_winner_deg','mt_population_deg','mt_resultant']
    fields += [f'projected_d{i}_hz' for i in range(8)]+[f'mt_d{i}' for i in range(8)]
    fields += [f'v1_c{i}_hz' for i in range(84)]
    writer = csv.DictWriter(metrics_stream,fieldnames=fields)
    writer.writeheader()
    total_mt = np.zeros(8,np.float64)
    total_projected = np.zeros(8,np.float64)
    evaluated,frames,v1_seconds,downstream_seconds = 0,0,0.0,0.0
    started = perf_counter()
    success = False
    # File outputs remain invalid/incomplete without run_report.json on failure.
    try:
        for index,frame in iter_stimulus(stimulus,max_frames):
            op.sync()
            begin = perf_counter()
            rates = model.step(frame,index*frame_ms/1000)
            op.sync()
            v1_s = perf_counter()-begin
            begin = perf_counter()
            if op.torch:
                with op.torch.inference_mode():
                    mt = mt_model.step(rates)
                    estimate = mst.step(mt,frame_ms/1000) if mst else None
            else:
                mt = mt_model.step(rates)
                estimate = mst.step(mt,frame_ms/1000) if mst else None
            op.sync()
            downstream_s = perf_counter()-begin
            # Summaries are small device reductions. Full host copies only for requested files.
            roi = (...,slice(border,h-border),slice(border,w-border))
            if op.torch:
                mt_mean = op.host(mt['activity'][roi].mean(dim=(-2,-1)))
                projected_mean = op.host(mt['direction_rates'][roi].mean(dim=(-2,-1)))
                raw_mean = op.host(rates[roi].mean(dim=(-2,-1))).reshape(-1)
            else:
                mt_mean = mt['activity'][roi].mean(axis=(-2,-1))
                projected_mean = mt['direction_rates'][roi].mean(axis=(-2,-1))
                raw_mean = rates[roi].mean(axis=(-2,-1)).reshape(-1)
            direction_stream.write(op.host(mt['direction_rates']).astype('<f4',copy=False).tobytes())
            if activity_stream:
                activity_stream.write(op.host(mt['activity']).astype('<f4',copy=False).tobytes())
            if raw:
                raw.append(op.host(rates))
            if estimate_stream:
                estimate.update({'input_timestamp_s':index*frame_ms/1000,
                                 'estimate_timestamp_s':(index+1)*frame_ms/1000})
                estimate_stream.write(json.dumps(estimate,allow_nan=False)+'\n')
            summary = _direction_summary(mt_mean,projection.directions_rad)
            row = {'frame':index,'timestamp_s':index*frame_ms/1000,
                   'available_timestamp_s':(index+1)*frame_ms/1000,
                   'v1_s':v1_s,'mt_mstd_s':downstream_s,'mt_evidence':mt['evidence'],
                   'mt_winner_deg':summary['winner_deg'],'mt_population_deg':summary['population_deg'],
                   'mt_resultant':summary['resultant']}
            row.update({f'projected_d{i}_hz':float(v) for i,v in enumerate(projected_mean)})
            row.update({f'mt_d{i}':float(v) for i,v in enumerate(mt_mean)})
            row.update({f'v1_c{i}_hz':float(v) for i,v in enumerate(raw_mean)})
            writer.writerow(row)
            if index>=warmup_frames:
                total_mt += mt_mean
                total_projected += projected_mean
                evaluated += 1
            frames += 1
            v1_seconds += v1_s
            downstream_seconds += downstream_s
            if progress_every and (frames==1 or frames%progress_every==0 or frames==count):
                print(f'[V1-MT] {frames}/{count}, elapsed={perf_counter()-started:.2f}s',flush=True)
        success = True
    finally:
        direction_stream.close()
        metrics_stream.close()
        if activity_stream:
            activity_stream.close()
        if estimate_stream:
            estimate_stream.close()
        if raw:
            raw.close(success)
    elapsed = perf_counter()-started
    mt_mean = total_mt/max(evaluated,1)
    check = _direction_summary(mt_mean,projection.directions_rad)
    expected = expected_direction_deg
    error = abs((check['winner_deg']-expected+180)%360-180) if expected is not None and check['winner_deg'] is not None else None
    population_error = abs((check['population_deg']-expected+180)%360-180) if expected is not None and check['population_deg'] is not None else None
    check.update({'expected_deg':expected,'error_deg':error,
                  'population_error_deg':population_error,
                  'passed':error is not None and error<=22.5 and evaluated>0})
    dump_json(directory/'arrays.json',{'dtype':'float32-le','layout':'TDYX','frames':frames,
              'shape':[frames,8,h,w],'directions_rad':list(projection.directions_rad),
              'direction_rates.f32':'Hz','mt_activity.f32':'dimensionless' if save_mt else 'not_saved'})
    if spec:
        # Reuse the existing direction-recording protocol without duplicate binary storage.
        metadata = {'schema':'v1-rate-v1','encoding':'direction_rates','dtype':'float32-le',
                    'layout':'TDYX','units':'Hz','coordinates':'x-right_y-down',
                    'timestamp_reference':'frame_start','sample_hold':'until_next_timestamp',
                    'width':w,'height':h,'directions_rad':list(projection.directions_rad),
                    'intrinsics_px':intrinsics,'timestamps_s':[i*frame_ms/1000 for i in range(frames)],
                    'provenance':{**provenance,'last_frame_duration_s':frame_ms/1000}}
        # Standard reader requires rates.f32 in the same folder.
        (directory/'direction_rates.f32').rename(directory/'rates.f32')
        dump_json(directory/'metadata.json',metadata)
        arrays = json.loads((directory/'arrays.json').read_text(encoding='utf-8'))
        arrays['rates.f32'] = arrays.pop('direction_rates.f32')
        dump_json(directory/'arrays.json',arrays)
    report = {'schema':'python-v1-mt-run-v1','complete':True,'frames':frames,
              'source':provenance,'projection_name':projection_name,
              'projection_config_sha256':file_sha256(projection_path),'radius':radius,'border':border,
              'warmup_frames':warmup_frames,'evaluated_frames':evaluated,
              'mt_mean_activity':mt_mean.tolist(),
              'projected_mean_hz':(total_projected/max(evaluated,1)).tolist(),
              'direction_check':check,'mstd_enabled':mst is not None,
              'wall_seconds':elapsed,'v1_seconds':v1_seconds,'mt_mstd_seconds':downstream_seconds,
              'wall_fps':frames/elapsed,'limitations':[
                  'direction check is an engineering criterion, not a full V1 equivalence test',
                  'continuous rates do not reproduce a particular spike realization',
                  'no event/OMS path; no automatic image resize or delay compensation']}
    dump_json(directory/'run_report.json',report)
    return report


def compare_raw(reference,candidate,output,projection_path,projection_name='motionProj1',
                atol=5e-4,rtol=5e-4,radius=1):
    if any(not isfinite(x) or x<0 for x in (atol,rtol)):
        raise ValueError('finite nonnegative tolerances required')
    ref,can = raw_info(reference),raw_info(candidate)
    for key in ('width','height','frames','frame_interval_s'):
        if ref[key]!=can[key]:
            raise ValueError(f'comparison metadata mismatch: {key}')
    for key in ('stimulus_sha256','source_sha256'):
        a,b = ref.get('provenance',{}).get(key),can.get('provenance',{}).get(key)
        if not a or a!=b:
            raise ValueError(f'comparison requires identical {key}')
    directory = Path(output)
    directory.mkdir(parents=True,exist_ok=False)
    projection = load_projection(projection_path,projection_name)
    mt = DirectionMT(ArrayOps(),projection,ref['width'],ref['height'],radius)
    total_sq,total_ref_sq,total_n,max_abs,failed = 0.0,0.0,0,0.0,0
    channel_sq = np.zeros((3,28),np.float64)
    stage_max = {'projected':0.0,'pooled':0.0,'mt':0.0}
    stage_pass = dict.fromkeys(stage_max,True)
    with (directory/'frames.csv').open('x',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream,fieldnames=['frame','raw_max_abs','raw_rmse','raw_failed_values',
                                                  'projected_max_abs','pooled_max_abs','mt_max_abs'])
        writer.writeheader()
        for i,(a,b) in enumerate(zip(iter_raw(reference),iter_raw(candidate),strict=True)):
            delta = b.astype(np.float64)-a
            squared = delta*delta
            count = int((np.abs(delta)>atol+rtol*np.abs(a)).sum())
            failed += count
            maximum = float(np.abs(delta).max())
            max_abs = max(max_abs,maximum)
            total_sq += float(squared.sum())
            total_ref_sq += float((a.astype(np.float64)**2).sum())
            total_n += a.size
            channel_sq += squared.sum(axis=(-2,-1))
            ma,mb = mt.step(a),mt.step(b)
            row = {'frame':i,'raw_max_abs':maximum,'raw_rmse':float(np.sqrt(squared.mean())),
                   'raw_failed_values':count}
            for name,key in (('projected','direction_rates'),('pooled','pooled_rates'),('mt','activity')):
                difference = np.abs(mb[key].astype(np.float64)-ma[key])
                value = float(difference.max())
                stage_max[name] = max(stage_max[name],value)
                stage_pass[name] = stage_pass[name] and bool((difference<=atol+rtol*np.abs(ma[key])).all())
                row[name+'_max_abs'] = value
            writer.writerow(row)
    with (directory/'channels.csv').open('x',newline='',encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['scale','filter','rmse'])
        for s in range(3):
            for f in range(28):
                writer.writerow([s,f,float(np.sqrt(channel_sq[s,f]/(ref['frames']*ref['width']*ref['height'])))])
    report = {'schema':'v1-continuous-comparison-v1','passed':failed==0 and all(stage_pass.values()),
              'raw_passed':failed==0,'atol':atol,'rtol':rtol,'frames':ref['frames'],
              'raw_failed_values':failed,'raw_values':total_n,'raw_max_abs':max_abs,
              'raw_rmse':float(np.sqrt(total_sq/total_n)),
              'relative_l2':float(np.sqrt(total_sq/total_ref_sq)) if total_ref_sq else None,
              'stage_max_abs':stage_max,'stage_passed':stage_pass,
              'reference':ref,'candidate':can,'projection_name':projection_name,
              'projection_config_sha256':file_sha256(projection_path),
              'limitations':['finite precision engineering tolerances, not biological validation',
                             'same frame indices; no fitted gain or time shifting',
                             'MSTd heading accuracy requires separate motion truth']}
    dump_json(directory/'comparison.json',report)
    return report
