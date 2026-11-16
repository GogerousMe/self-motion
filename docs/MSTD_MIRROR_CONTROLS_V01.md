# 下一轮：±7°严格镜像、多神经种子对照

日期：2026-11-09。当前七段基线已经完成，六段达到目标，云点右偏+7°的MAE为2.867°。本轮先定位该偏移的稳定性，最终检查对象仍为MSTd航向；同时保留V1/MT分层响应。模型配置、E/I参数、模板和几何门控沿用冻结v01。

## 1 已准备的输入

```text
input/stimuli/selfmotion/mstd_mirror_s0211__80x80__f000080__dt50ms__2cases__v01/
  cloud_heading_left7.dat
  cloud_heading_right7.dat
  cloud_heading_left7.truth.jsonl
  cloud_heading_right7.truth.jsonl
  intrinsics.json
  manifest.json
  artifact.json
  verification.json
  preview/
    cloud_heading_left7.mp4
    cloud_heading_right7.mp4
    left_minus7__right_plus7.mp4
```

每个输入80×80、80帧、50ms/帧、4秒；25字节VisualStimulus头，单通道灰度TYX像素。仍是图像DAT，不将点坐标、深度或真值输入网络。

使用新场景种子211生成右偏+7°云点序列，再将每一帧严格水平反射得到左偏−7°。此前同种子分别生成左右运动并不能保证两段像素互为镜像；这次满足：

```text
left[t,y,x] == right[t,y,79-x]
```

真值同时反射：u_left=79−u_right，du_left=−du_right，平移横向速度/位置/航向变号；纵向像素位置、深度、ID、帧号、时间保持。末帧未知的后续光流保持null。软点中心在边界附近时不裁剪坐标或重新生成点，以保持像素和真值对应。

两段像素及逐点真值已核验；MP4均为H264、80帧、20fps、4秒。并排预览左侧为−7°、右侧为+7°，只用于观察，不是模型输入。

这是针对已知问题设计的开发对照；一个新场景配三个神经种子不能替代后续独立场景的最终验收。

## 2 需要在服务器运行什么

固定两段图像，神经Poisson种子使用43、44、45，每个方向运行三次，共6次、480帧。每次独立初始化，避免上一段的状态残留干扰镜像比较。

同一神经种子不会强制产生镜像脉冲样本，因为镜像后像素/神经元位置不同。因此不要求逐帧结果严格互为相反数，而是比较方向误差、平均偏移及其随随机种子的变化。

本轮首先只运行这6次，聚焦航向偏移。零输入、静态纹理及运动停止的残留对照作为下一小组，现有静止实验继续保留为基线。

## 3 同步文件

服务器已有此前运行成功的完整v01模型、配置、冻结文件及run_model.py。同步以下新增/更新文件即可，也可以按既有方式直接复制源码：

```text
src/mstd/mirror_controls.py
tools/prepare_mstd_mirror_controls.py
tools/run_mstd_mirror_controls.py
examples/inspect_mstd_cloud_holdout.py
tests/test_mirror_controls.py
tests/test_mirror_batch_summary.py
docs/MSTD_MIRROR_CONTROLS_V01.md
input/stimuli/selfmotion/mstd_mirror_s0211__80x80__f000080__dt50ms__2cases__v01/
```

复用已有：

```text
tools/run_selfmotion_baseline.py
examples/check_mstd_validity.py
configs/model_python_spiking_geometry_v01.json
experiments/validation/20261109__mstd_geometry_gate_80x80_f000080__py_rate_cpu__r01/frozen_validity.json
```

批量脚本会核对DAT、manifest、镜像关系、索引、内参和冻结参数。服务器索引若缺少这个新套件，会在实际运行前追加已核验的刺激/套件登记；保留服务器原有记录。已有索引若哈希冲突则报错。无需用本机experiments/registry.csv覆盖服务器索引。

## 4 实际命令

在服务器的代码根目录：

```bash
conda activate mstd
cd /data/workspace/WYQ/mstd/code
python tools/run_mstd_mirror_controls.py --backend torch --device cuda
```

将依次运行左/右各三次，自动核验每次GPU判据，汇总MSTd结果并执行保存响应的分层分析。不需要自行重复输入六条命令。参数或设备不符合要求时不会静默回退CPU。

如需先核对输入与执行计划：

```bash
python tools/run_mstd_mirror_controls.py --plan-only
```

该模式只读文件、打印六条结构化命令，不运行模型或写入登记。

依赖沿用现有NumPy、CUDA PyTorch；运行已经生成的DAT不需要ffmpeg。仅重新生成输入/预览时需要ffmpeg和ffprobe：

```bash
python tools/prepare_mstd_mirror_controls.py
```

生成器拒绝覆盖已存在套件；既然新套件已经复制到服务器，就不必再运行生成器。

## 5 输出位置与查看方法

批次目录自动命名，例如：

```text
experiments/validation/<time>__mirror7_s0211_80x80_f000480__py_spike_cuda__r01/
  REPORT.md
  summary.json
  summary.csv
  batch.json
  artifact.json
  cloud_heading_left7_s0043.log
  ...其余五次运行日志...
  layer_analysis.log
```

终端完成时打印 `BATCH_OUTPUT:`、`BATCH_REPORT:`、`LAYER_ANALYSIS:`。单次原生响应继续保存在experiments/validation下的独立运行目录，含V1/MT/MSTd监视器、实际脉冲和配置快照；每次运行还产生对应gate_check报告。batch.json列出六个准确路径。

分层分析位于：

```text
outputs/comparisons/selfmotion/<batch_id>__layers__v01/
```

比较V1线性投影、MT实际帧内放电率方向场、MT指数轨迹和MSTd竞争结果。V1线性投影是诊断对照，不宣称与神经E/I网络等价；分层时间窗也不同。该脚本新增了可选suite路径，并用float64累积模板审计分数，避免长float32求和引入额外核验误差，没有修改模型计算。

若中断或失败，批次标记partial，已有响应和日志保留。执行完成与诊断目标通过分别记录；summary.json中的all_six_targets_passed才是六个条件是否都达到既定目标。

## 6 本轮验证什么

主指标使用预热后的60帧，保留原5°航向网格，不将诊断插值当模型结果：

1. MSTd有效率是否≥95%，未门控MAE是否≤2.5°。
2. 左右分别偏向−5°/+5°，还是某一侧持续偏向更远模板。
3. 三个种子下的有符号误差、95分位和最大误差是否相近。
4. 镜像偏移：mean(左预测+右预测)/2是否持续偏向同一侧。该值是描述性指标，不是概率或显著性检验。
5. 若存在稳定偏移，它最早出现在V1投影、MT实际脉冲、指数轨迹还是MSTd竞争。

若偏移随种子明显变化，先评估脉冲噪声和读出的时间稳定性；若镜像对照在多个种子下仍保持同向偏移，进一步检查对应层的方向通道/驱动平衡。不能仅凭三个种子直接认定某个参数错误。

本轮不调整门控阈值，也不通过只统计有效帧来掩盖航向误差。之后再决定最小机制改动，并用旧七段回归及新留出场景验收。

## 7 本机准备核验

镜像像素、真值与相机投影关系、镜像偏移统计、批次汇总使用未门控误差和只读计划模式均有针对性检查；输入及预览已生成并核验。本机未运行新的六次CUDA模型，需由上述服务器命令执行。
