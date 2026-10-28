# 三阶段 MT 空间响应验证：服务器执行步骤

日期：2026-10-25。目的：比较同一帧的投影、局部池化和归一化是否保持合理空间结构与方向选择性，不把热图相似程度当作唯一准确率指标。

## 1. 上传文件

从本地code目录同步以下三个文件到服务器 `/data/workspace/wyq/mstd/code/` 的同名相对路径：

```text
src/mstd/mt.py
examples/visualize_mt_gratings.py
examples/compare_mt_stages.py
```

新脚本依赖相邻的visualize_mt_gratings.py，两者都需上传。若要运行新增回归测试，同时同步tests/。原有MT向量与证据数值不变，仅增加可选的池化率导出。Python仍需>=3.10，无额外Python包，需ffmpeg及libx264。

## 2. 配对与环境检查

```bash
conda activate mstd
cd /data/workspace/wyq/mstd/code
export PYTHONPATH="$PWD/src"
python --version
ffmpeg -version
python examples/compare_mt_stages.py --help
```

输入刺激可直接指向CARLsim工程，不必复制。注意指定的 `selfmotion_8directionsx3speed.dat` 是否确实是之前生成 `trail/8directions/spk_V1.dat` 的原输入，及是否与所用manifest配套。之前清单为四速度、4800帧；文件名x3speed不能证明内容相同或不同。

先执行轻量检查（只读输入和文件头，不遍历19亿事件）：

```bash
python - <<'PY'
import json
from pathlib import Path
from mstd.calibration import inspect_stimulus, file_sha256
from mstd.carlsim import read_header
p = Path('/data/workspace/xwz/git/CARLsim6/projects/v1mt_xwz/input/selfmotion/selfmotion_8directionsx3speed.dat')
m = Path('srcipts/bw_grating_output/sequence_20261019_134626_053/manifest.json')
r = json.loads(m.read_text(encoding='utf-8-sig'))
print('actual stimulus:', inspect_stimulus(p))
print('manifest:', r['width'], r['height'], r['runs']['frames'], r['frame_ms'], r['runs']['duration_ms'])
print('stimulus SHA256 matches:', file_sha256(p) == r['runs']['stimulus_sha256'])
print('V1 header:', read_header('trail/8directions/spk_V1.dat'))
PY
```

清单配对、尺寸或哈希不一致时先停止，找原刺激对应清单/分段CSV及采集日志，不要修改哈希或帧数强行运行。如果确实使用三速度刺激，需要根据真实条件顺序和时长建立它自己的清单及分段表。

## 3. 短预览：先300帧

```bash
python examples/compare_mt_stages.py \
  --manifest srcipts/bw_grating_output/sequence_20261019_134626_053/manifest.json \
  --stimulus /data/workspace/xwz/git/CARLsim6/projects/v1mt_xwz/input/selfmotion/selfmotion_8directionsx3speed.dat \
  --spikes trail/8directions/spk_V1.dat \
  --projection-config configs/v1_projection.json \
  --projection motionProj1 \
  --radius 1 \
  --max-frames 300 \
  --rate-max 300 --activity-max 1 --colormap jet \
  --progress-every 20 \
  --output outputs/mt_stage_preview01
```

输出目录必须不存在。若清单是原四速度4800帧版本，前300帧覆盖右、右下两种方向及对应黑场，总计15秒。只读这部分原始响应，不需要先处理完整19亿事件；前缀预览不是整文件验证。换输入清单后以其真实分段为准。

屏幕四行：

1. 原输入灰度图、方向文字及箭头、速度、时间窗。
2. STAGE 1：V1经28→8投影的空间方向率，Hz。
3. STAGE 2：默认3×3局部池化后的空间方向率，Hz。
4. STAGE 3：八方向归一化后MT空间活动，无量纲。

各响应行均按右0°、右下45°、下90°、左下135°、左180°、左上225°、上270°、右上315°排列。黄色表示输入期望方向，青色为该阶段全图均值最强方向；黑场没有期望方向。V1原始84通道不展示。

两个Hz阶段共用固定0～300色标，归一化阶段固定0～1；不能跨不同单位仅凭颜色比较绝对强度。rate-max只是显示设置，不改变模型。jet低值为蓝、高值为红。

## 4. 查看并反馈哪些文件

```text
outputs/mt_stage_preview01/
  mt_stages.mp4                   # 四行同步视频，1440x1230
  stage_segment_001.png           # 右方向示例，原清单下为1秒处
  stage_segment_003.png           # 右下示例，原清单下为8.5秒处
  stage_segment_*.png             # 各段稳定位置和第一帧
  stage_statistics.csv            # 每帧每阶段的方向均值/极值/空间标准差/色标饱和比例
  stages.f32                     # 三阶段空间缓存
  stages.json                    # 缓存单位、方向布局、哈希、是否完整验证
  video_config.json
```

可下载MP4在本机播放器查看。首次建议返回右、右下的PNG、stage_statistics.csv及video_config.json，不用上传原始V1或大缓存。

## 5. 如何判读

- 投影行是否已经有空间结构：若有条纹而池化后稍平滑，符合邻域平均预期；若投影已经均匀，不应期待后续重新恢复条纹。
- 池化行：空间结构应由局部邻域传播，默认半径1。边界使用实际有效邻域，不跨整个图像广播。
- 归一化行：同一位置的八方向共同决定输出，可能压缩明暗对比、使强响应趋于饱和，不要求复制输入亮度。
- 正确方向及相邻方向响应应与刺激一致；不能出现图像翻转、始终固定一个方向，或用期望标签强行点亮。
- 看黑场开始后的衰减及新运动的启动，而不是只看静态截图。
- CSV中的 `std_at_expected` 为期望方向全图标准差，用于量化空间强弱差异；它不是准确率，不能要求越大越好。
- `fraction_at_or_above_color_max` 很高时说明色标可能截断细节，应从缓存调大rate-max重绘；视觉整块红不等于数值完全一致。
- 该可视化均值包括图像边缘，之前方向报告排除了4像素边界，二者均值不必完全相同。

完整方向调谐准确率仍结合先前report/frames分析；视频用于定位处理阶段和空间问题，不能证明孔径问题已经解决或MSTd航向正确。

## 6. 缓存重绘：不重读V1

若0～300Hz过亮/过暗，只调整显示：

```bash
python examples/compare_mt_stages.py \
  --manifest srcipts/bw_grating_output/sequence_20261019_134626_053/manifest.json \
  --stimulus /data/workspace/xwz/git/CARLsim6/projects/v1mt_xwz/input/selfmotion/selfmotion_8directionsx3speed.dat \
  --cache outputs/mt_stage_preview01/stages.json \
  --rate-max 500 --activity-max 1 --colormap jet \
  --output outputs/mt_stage_preview_recolor01
```

改radius或投影参数会改变模型，不能靠同一缓存完成，需重新处理V1。重绘只改变色标/播放设置。新旧缓存不能混用：compare脚本使用stages.json（三阶段），旧单行脚本使用mt_activity.json（单阶段）。

## 7. 完整视频与存储

短预览确认后，使用第3节命令，删去 `--max-frames 300` 并换输出目录 `outputs/mt_stage_full01`。默认完整20fps、240秒。整个过程只遍历一次V1事件流。

三阶段float32缓存大小：80×80×8×3×4800×4 = 2949120000字节，约2.75GiB；预览300帧约176MiB。另预留MP4和PNG空间。流式处理，不把所有帧读入内存；首次计算和编码仍可能较慢，依据进度ETA等待。

## 8. 本机验证边界

34项测试通过，新增局部脉冲测试确认：50ms单脉冲得到20Hz；3×3池化只扩散到局部邻域，远处保持零；半径0保留投影结构；原有MT向量/证据不变。三阶段人工空间条纹视频已成功编码，缓存重绘得到相同画面。

人工夹具用于验证程序，不是服务器真实响应。MATLAB原图若来自另一个C++MT网络，数值单位和动力学不同，不应要求Python图与其逐像素相同。
