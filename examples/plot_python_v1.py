"""Render V1/MT diagnostic plots from an offline run; requires matplotlib."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording')
    parser.add_argument('--frame',type=int,default=30)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(args.recording)
    report=json.loads((root/'run_report.json').read_text(encoding='utf-8'))
    arrays=json.loads((root/'arrays.json').read_text(encoding='utf-8'))
    if not report.get('complete') or not 0<=args.frame<arrays['frames']:
        raise ValueError('completed run and valid zero-based frame index required')
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    angles=np.degrees(arrays['directions_rad'])%360
    order=np.argsort(angles)
    labels=[f'{angles[i]:.0f} deg' for i in order]
    data=np.genfromtxt(root/'frames.csv',delimiter=',',names=True)
    data=np.atleast_1d(data)
    fig,axes=plt.subplots(3,1,figsize=(12,9),sharex=True,layout='constrained')
    for i in order:
        axes[0].plot(data['timestamp_s'],data[f'projected_d{i}_hz'],label=f'{angles[i]:.0f} deg')
        axes[1].plot(data['timestamp_s'],data[f'mt_d{i}'])
    axes[0].set_ylabel('Projected rate (Hz)');axes[0].legend(ncol=4)
    axes[1].set_ylabel('MT activity')
    axes[2].plot(data['timestamp_s'],data['mt_evidence'])
    axes[2].set_ylabel('MT evidence');axes[2].set_xlabel('Input time (s)')
    fig.savefig(output/'time_courses.png',dpi=160);plt.close(fig)
    shape=tuple(arrays['shape'])
    direction_file='rates.f32' if (root/'metadata.json').exists() else 'direction_rates.f32'
    for filename,title,unit in ((direction_file,'Projected direction rates','Hz'),
                                 ('mt_activity.f32','Normalized MT activity','dimensionless')):
        if not (root/filename).exists():continue
        value=np.memmap(root/filename,mode='r',dtype='<f4',shape=shape)[args.frame]
        fig,axes=plt.subplots(2,4,figsize=(14,7),layout='constrained')
        vmax=max(float(value.max()),1e-12)
        for ax,i,label in zip(axes.flat,order,labels):
            artist=ax.imshow(value[i],cmap='viridis',vmin=0,vmax=vmax,origin='upper')
            ax.set_title(label);ax.set_xticks([]);ax.set_yticks([])
        fig.colorbar(artist,ax=list(axes.flat),label=unit)
        fig.suptitle(f'{title}; frame {args.frame} (shared color scale)')
        fig.savefig(output/(filename.replace('.f32','')+'.png'),dpi=160);plt.close(fig)
    raw_meta=root/'raw_v1/metadata.json'
    if raw_meta.exists():
        from mstd.v1_recording import raw_info
        meta=raw_info(root/'raw_v1')
        value=np.memmap(root/'raw_v1/rates.f32',mode='r',dtype='<f4',
                        shape=(meta['frames'],3,28,meta['height'],meta['width']))[args.frame]
        b=report['border'];roi=value[...,b:meta['height']-b,b:meta['width']-b]
        fig,ax=plt.subplots(figsize=(12,3),layout='constrained')
        artist=ax.imshow(roi.mean(axis=(-2,-1)),aspect='auto',cmap='viridis')
        ax.set_xlabel('Space-time filter index');ax.set_ylabel('Scale block');ax.set_yticks([0,1,2])
        fig.colorbar(artist,ax=ax,label='Rate (Hz)')
        fig.savefig(output/'v1_channel_means.png',dpi=160);plt.close(fig)
    print(output)


if __name__=='__main__':main()
