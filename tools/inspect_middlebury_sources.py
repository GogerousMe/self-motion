"""Audit local Middlebury frames, duplicate pairs and public flow references."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import struct
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mstd.calibration import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(r'D:\A_Gorger\program\Datasets'))
    args = parser.parse_args()
    rows = []
    for group, child in [('other-color-allframes', 'other-data'), ('eval-color-allframes', 'eval-data')]:
        for directory in sorted((args.source / group / child).iterdir()):
            if not directory.is_dir():
                continue
            paths = sorted(directory.glob('frame*.png'))
            if not paths:
                raise ValueError(f'no source frames in {directory}')
            hashes = {path.name: file_sha256(path) for path in paths}
            sizes = []
            for path in paths:
                with Image.open(path) as image:
                    image.load()
                    sizes.append(image.size)
            if len(set(sizes)) != 1:
                raise ValueError(f'changing source geometry in {directory}')
            pair_root = args.source / group.replace('allframes', 'twoframes') / child / directory.name
            duplicates = {}
            for name in ('frame10.png', 'frame11.png'):
                path = pair_root / name
                duplicates[name] = path.is_file() and file_sha256(path) == hashes.get(name)
            flow_path = args.source / 'other-gt-flow/other-gt-flow' / directory.name / 'flow10.flo'
            row = dict(group=group, sequence=directory.name, source_directory=str(directory),
                       frames=len(paths), width=sizes[0][0], height=sizes[0][1],
                       frame_names=[path.name for path in paths], image_sha256=hashes,
                       twoframe_pair_identical=duplicates, public_flow=flow_path.is_file(),
                       camera_pose_gt_available_in_supplied_bundles=False,
                       source_frame_interval_s=None, current_v1_history_frames=9,
                       full_native_history_available=len(paths) >= 9)
            if flow_path.is_file():
                raw = flow_path.read_bytes()
                if len(raw) < 12:
                    raise ValueError(f'truncated flow file {flow_path}')
                magic, width, height = struct.unpack('<fii', raw[:12])
                if (magic != 202021.25 or min(width, height) < 1
                        or len(raw) != 12 + width * height * 8 or (width, height) != sizes[0]):
                    raise ValueError(f'invalid flow header/geometry {flow_path}')
                flow = np.frombuffer(raw, dtype='<f4', offset=12).reshape(height, width, 2)
                valid = np.isfinite(flow).all(axis=2) & (np.abs(flow) <= 1e9).all(axis=2)
                magnitude = np.linalg.norm(flow[valid].astype(np.float64), axis=1)
                row.update(flow_file=str(flow_path), flow_sha256=file_sha256(flow_path),
                           flow_units='source pixels per frame10-to-frame11 pair',
                           valid_flow_fraction=float(valid.mean()),
                           displacement_median_px=float(np.median(magnitude)),
                           displacement_p95_px=float(np.percentile(magnitude, 95)))
            interpolation = args.source / 'other-gt-interp/other-gt-interp' / directory.name
            row['interpolation_gt_files'] = [str(path) for path in sorted(interpolation.glob('*.png'))]
            rows.append(row)

    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    rid = stamp + '__middlebury_source_audit__v01'
    out = ROOT / 'experiments/validation' / rid
    out.mkdir(parents=True, exist_ok=False)
    public = [row for row in rows if row['public_flow']]
    result = dict(schema='middlebury-source-audit-v1', sequences=rows,
                  public_flow_sequences=len(public),
                  public_flow_eight_frame_sequences=sum(row['frames'] == 8 for row in public),
                  public_flow_two_frame_sequences=sum(row['frames'] == 2 for row in public),
                  recommended_mt_sequences=['RubberWhale', 'Hydrangea', 'Urban2'],
                  next_protocol_requirements=[
                      'Use the allframes source; twoframes contains the same central pair.',
                      'Record synthetic boundary extension explicitly if testing the current nine-frame V1.',
                      'Do not infer native frame duration from the archive name.',
                      'Align the delayed V1/MT output to flow10 (frame10 to frame11); no post-hoc best lag.',
                      'Scale flow u and v with image geometry and preserve the unknown-flow mask.',
                      'Measure local 2D direction error and coverage with frozen parameters.',
                      'The supplied bundles contain image-pair flow GT; heading/pose scoring needs additional camera-motion GT.'])
    (out / 'inventory.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    lines = ['# Middlebury本地数据检查', '',
             '本轮完成原图、中央两帧重复关系、FLO头/尺寸/无效值及文件哈希检查。', '',
             '| 序列 | 图像帧数 | 尺寸 | 公开光流真值 | 有效光流像素比例 |',
             '|---|---:|---|---|---:|']
    for row in rows:
        valid = f"{row['valid_flow_fraction']:.2%}" if row['public_flow'] else '—'
        lines.append(f"| {row['sequence']} | {row['frames']} | {row['width']}×{row['height']} | "
                     f"{'有' if row['public_flow'] else '未提供'} | {valid} |")
    lines.extend(['', '## 本轮使用建议', '',
                  '优先选择RubberWhale、Hydrangea、Urban2开展MT局部方向专项。',
                  '公开光流共8组，其中6组8帧、2组2帧。Dimetrodon与Venus均只有2帧。',
                  'twoframes中的中央两帧已经逐文件与allframes核对。',
                  '当前V1使用9帧历史，所有本地序列都需要显式处理时间边界后再定量评价。',
                  '预热重复帧只能表达边界假设，不能增加真实观测。需要保留原始帧号和延迟对齐。',
                  'flow10对应frame10到frame11的二维像素位移。other-gt-interp提供中间图像真值。',
                  '本轮先完成数据检查，Middlebury模型刺激生成和GPU响应测试待短序列协议确定后执行。', '',
                  '[官方数据与格式说明](https://vision.middlebury.edu/flow/data/)', ''])
    (out / 'SOURCE_REVIEW.md').write_text('\n'.join(lines), encoding='utf-8')
    artifact = dict(schema='mstd-artifact-v1', id=rid, kind='validation', status='completed',
                    stimulus_ids=[], parent_run_ids=[], input_paths=[str(args.source)],
                    source_hashes={'tools/inspect_middlebury_sources.py': file_sha256(Path(__file__))},
                    notes='Read-only image/flow integrity audit and applicability review; no model run.')
    (out / 'artifact.json').write_text(json.dumps(artifact, indent=2), encoding='utf-8')
    registry = ROOT / 'experiments/registry.csv'
    with registry.open(encoding='utf-8-sig', newline='') as stream:
        fields = csv.DictReader(stream).fieldnames
    with registry.open('a', encoding='utf-8', newline='') as stream:
        csv.DictWriter(stream, fieldnames=fields).writerow(dict(id=rid, kind='validation',
            path=out.relative_to(ROOT).as_posix(), status='completed', notes=artifact['notes']))
    print(json.dumps(dict(output=str(out), public_flow=len(public),
                         public_flow_eight_frames=result['public_flow_eight_frame_sequences'],
                         all_central_pairs_match=all(all(row['twoframe_pair_identical'].values()) for row in rows)), indent=2))


if __name__ == '__main__':
    main()
