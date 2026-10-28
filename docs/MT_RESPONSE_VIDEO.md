# 输入光栅与八方向 MT 响应同步视频

## 放电频率热图模式（2026-10-25补充）

现在支持 `--colormap jet`，以及以下三种严格区分的数据：

| 输入与选项 | 图中数据 | 单位 |
|---|---|---|
| `--spikes V1.dat`（默认） | Python MT局部池化、归一化后的方向活动 | 无量纲 |
| `--spikes V1.dat --response projected-hz` | V1经28→8带符号投影后的方向率，尚未经过MT池化归一化 | 投影Hz，不是实际MT放电率 |
| `--mt-spikes MT.dat --mt-directions-deg ...` | 实际MT脉冲按每窗计数除以秒数 | 真实脉冲统计Hz |

实际MT模式要求CARLsim SpikeMonitor v0.2、W×H×8，W/H与当前输入一致，编号布局为x最快、y其次、方向平面最后。必须显式填写存储的八个平面分别对应的物理运动角度，不能猜测通道顺序；例如源代码确认顺序为右、右上、上、左上、左、左下、下、右下时，参数为 `--mt-directions-deg 0 315 270 225 180 135 90 45`。若每方向保存在单独文件、深度不是8、存在独立E/I群体或输出降采样，需要按实际文件头和连接定义适配，不要补零冒充八方向。

Hz = count×1000/window_ms。50ms窗内1个脉冲为20Hz；MATLAB示例FrameDur=1000时1个脉冲为1Hz。不能把本次的50ms窗随意改为1000ms，再沿用逐帧输入对应。无脉冲就是0Hz，不把静默方向估计成有效运动。

与MATLAB的 `CLim=[0,20]` 相同，使用 `--vmax 20` 固定色标上限；若很多神经元超过20Hz会被截为同色，建议先使用 `--vmax 100` 或参考真实响应分布调整，全程八方向共用同一色标。jet模式的0为深蓝色，不是黑色。

Python MT当前是活动率模型，并没有生成MT脉冲DAT；不能把归一化活动改个单位就称作Hz。如需当前模型输出的可视化，使用默认模式；如需旧C++MT网络的实际放电频率，必须提供它的真实MT脉冲文件。

实际MT文件示例（路径和通道角度必须替换为实际已核对值）：

```bash
python examples/visualize_mt_gratings.py \
  --manifest /path/to/matching/manifest.json \
  --stimulus /data/workspace/xwz/git/CARLsim6/projects/v1mt_xwz/input/selfmotion/selfmotion_8directionsx3speed.dat \
  --mt-spikes /path/to/ACTUAL_MT_SPIKES.dat \
  --mt-directions-deg 0 315 270 225 180 135 90 45 \
  --colormap jet --vmax 100 \
  --output outputs/actual_mt_hz_video01
```

若只有当前V1脉冲文件，也可增加 `--response projected-hz --colormap jet --vmax 100`，绘制清楚标注为投影率的八方向热图；没有真实MT脉冲参与这种模式。视频不显示V1原始84通道。

缓存记录response_kind与units；从缓存重绘保持原单位，不能通过设置--vmax将归一化活动变成Hz。所有模式仍支持同一输入两行布局、PNG快照和MP4视频输出。

## 显示的内容

`examples/visualize_mt_gratings.py` 输出两行1440×640视频：

- 第一行：实际输入 DAT 灰度帧、帧号、响应时间窗、刺激速度、方向名称及箭头。45°为右下。黑场显示无方向，不延续上一段标签。
- 第二行：按0、45、90、135、180、225、270、315°顺序并排展示八个MT方向空间热图，依次为右、右下、下、左下、左、左上、上、右上。
- 黄色标记来自刺激清单的期望方向，青色标记来自实际输出的最大空间均值通道；两者独立计算，不会强制一致。
- 每个面板附有全图均值。为了兼容没有中文字体的Ubuntu服务器，视频使用英文方向标签和箭头；RIGHT-DOWN即右下，LEFT-UP即左上。

八方向活动是现有MT编码中的“局部池化后，经过除法归一化的通道活动”。它们能够组合得到原有MT向量，不是直接显示V1原始脉冲，也不是MT神经元放电率Hz。新增导出选项不改变原有MT向量、证据或MSTd计算。

共用固定0～1色标，默认黑色低、红黄白色高，禁止各通道各帧分别自动拉伸。相邻方向可能共同活动，不能要求只有目标面板变亮。需要更亮时可统一使用`--vmax 0.6`，所有面板采用相同上限且大于上限的值会饱和。

时间对齐：第k幅刺激对应[k×50,(k+1)×50)ms内统计的响应；响应在窗口结束后才可获得，视频不代表窗口起点即可实时读出，不人为移动响应以追齐刺激。保留原始启动和黑场残留。

## 上传和环境

上传两个修改文件，保持目录结构：

```text
src/mstd/mt.py
examples/visualize_mt_gratings.py
```

Python仍为>=3.10、标准库，不需要Pillow、Matplotlib、NumPy或图形桌面。另需ffmpeg且支持libx264编码。已有ffmpeg时无需安装，可检查`ffmpeg -version`。Ubuntu如未安装，可由有权限的账户执行：

```bash
sudo apt update
sudo apt install ffmpeg
```

若ffmpeg不在PATH，使用`--ffmpeg /实际路径/ffmpeg`。视频不需要显示器或X11。

## 先生成短预览

若实际输入刺激位于CARLsim工程中，可添加 `--stimulus /实际路径/input.dat`，无需复制到Python项目。该文件仍须与manifest的尺寸、帧数及SHA256一致，分段标签和响应也须对应。仅改名或移动不影响哈希；内容不同则不能复用旧清单。视频仅展示输入与MT；`--spikes`仅作为后台MT计算来源，不展示V1。已有完整MT缓存时改用`--cache`，无需再次读取V1。

在服务器的code目录激活mstd环境并设置PYTHONPATH，先导出300帧（15秒，包含前两个方向右和右下及黑场）：

```bash
conda activate mstd
cd /data/workspace/wyq/mstd/code
export PYTHONPATH="$PWD/src"
python examples/visualize_mt_gratings.py \
  --manifest srcipts/bw_grating_output/sequence_20261019_134626_053/manifest.json \
  --spikes trail/8directions/spk_V1.dat \
  --projection-config configs/v1_projection.json \
  --projection motionProj1 \
  --max-frames 300 \
  --progress-every 20 \
  --output outputs/grating_mt_video_preview01
```

`--max-frames`只处理前缀，不代表完成原始整文件验证。默认不指定时导出全部4800帧。目录必须不存在；失败可能留下半成品视频或缓存，但不会生成有效缓存JSON，重试使用新目录。

## 完整4800帧视频

确认预览布局后执行：

```bash
python examples/visualize_mt_gratings.py \
  --manifest srcipts/bw_grating_output/sequence_20261019_134626_053/manifest.json \
  --spikes trail/8directions/spk_V1.dat \
  --projection-config configs/v1_projection.json \
  --projection motionProj1 \
  --output outputs/grating_mt_video01
```

输出：

- `mt_directions.mp4`：完整同步视频，默认20fps、240秒。
- `preview_segment_*.png`：各段预览图。
- `mt_activity.f32`和`mt_activity.json`：归一化八方向空间响应与元数据，用于后续重绘。按原投影顺序存储，绘图再重排为顺时针方向。
- `video_config.json`：色标、播放速度、时间语义、是否包含完整清单等信息。

全4800帧、80×80×8个float32缓存约983MB（937.5MiB）。另给视频、预览图预留空间。脚本逐帧写入，无需把整个缓存载入内存。首次生成仍需重读约19亿原始事件；现有frames.csv没有空间信息，不能还原热图。

## 使用缓存重绘

缓存完成后，可重新调色、加速播放，不必再次计算V1到MT：

```bash
python examples/visualize_mt_gratings.py \
  --manifest srcipts/bw_grating_output/sequence_20261019_134626_053/manifest.json \
  --cache outputs/grating_mt_video01/mt_activity.json \
  --vmax 0.6 \
  --playback-rate 2 \
  --output outputs/grating_mt_video_bright02
```

2倍播放为40fps、120秒，帧和方向仍一一对应。重绘校验缓存的清单、分段表与缓存文件哈希；不修改原响应，不会重复存储一份缓存。仅预览生成的缓存只能重绘该前缀，不能产生其后未计算的帧。

## 如何看结果

右下输入期间看45° RIGHT-DOWN面板是否强于相反方向，期望黄色标记和实际青色标记是否一致；查看切换后的启动瞬态与黑场消退，不用稳定期图替代整个时间过程。八方向相邻响应是调谐宽度的一部分。

热图与原CSV的数值定义不同：原CSV的八通道Hz是归一化之前的投影率，本视频为归一化之后的MT活动；原报告向量统计还排除了4像素边缘，本视频均值默认包含全图。因此不能把视频面板均值直接与CSV的Hz列对比。

本地视频验收使用人工脉冲及测试投影，仅验证方向排列、编码、时序、视频和缓存重绘功能；不能冒充本次真实V1视频。真实视频需在服务器使用原spk_V1.dat生成。
