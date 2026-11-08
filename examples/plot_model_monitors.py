"""Plot saved unified-model monitors. --frame means original input-frame index."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recording')
    parser.add_argument('--frame',type=int,default=0)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    root=Path(args.recording)
    report=json.loads((root/'model_report.json').read_text(encoding='utf-8'))
    if not report.get('complete'):raise ValueError('completed model run required')
    saved={}
    for path in (root/'monitors').glob('*/metadata.json'):
        meta=json.loads(path.read_text(encoding='utf-8'))
        if path.parent.name=='v1':
            shape=(meta['frames'],3,28,meta['height'],meta['width'])
            filename='rates.f32'
        else:
            if not meta.get('complete'):raise ValueError('incomplete monitor')
            shape=tuple(meta['shape']);filename=meta['file']
        binary=path.parent/filename
        if binary.stat().st_size!=int(np.prod(shape))*4:raise ValueError('monitor payload size mismatch')
        saved[path.parent.name]=(meta,np.memmap(binary,mode='r',dtype='<f4',shape=shape))
    if not saved:raise ValueError('no enabled monitor files found')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    for name,(meta,values) in saved.items():
        if name.startswith('mstd_'):
            fig,ax=plt.subplots(figsize=(11,4),layout='constrained')
            for k in range(values.shape[1]):
                ax.plot(meta['timestamps_s'],values[:,k],label=f'center x={meta["centers_px"][k][0]:.1f}px')
            ax.set_xlabel('Input time (s)');ax.set_ylabel(name.replace('_',' '))
            ax.legend(ncol=3,fontsize=8)
        else:
            indices=meta['sample_indices']
            if args.frame not in indices:raise ValueError(f'frame {args.frame} not saved by monitor {name}')
            value=values[indices.index(args.frame)]
            if name=='v1':
                fig,ax=plt.subplots(figsize=(12,3),layout='constrained')
                artist=ax.imshow(value.mean(axis=(-2,-1)),aspect='auto',cmap='viridis')
                ax.set_xlabel('Space-time filter');ax.set_ylabel('Scale block');ax.set_yticks([0,1,2])
                fig.colorbar(artist,ax=ax,label='Hz')
            else:
                angles=np.degrees(meta['directions_rad'])%360;order=np.argsort(angles)
                fig,axes=plt.subplots(2,4,figsize=(13,7),layout='constrained')
                for ax,i in zip(axes.flat,order):
                    artist=ax.imshow(value[i],origin='upper',cmap='viridis',vmin=0,vmax=max(float(value.max()),1e-12))
                    ax.set_title(f'{angles[i]:.0f} deg');ax.set_xticks([]);ax.set_yticks([])
                fig.colorbar(artist,ax=list(axes.flat),label=meta['units'])
            fig.suptitle(f'{name}; input frame {args.frame}')
        fig.savefig(output/(name+'.png'),dpi=150);plt.close(fig)
    print(output)


if __name__=='__main__':main()
