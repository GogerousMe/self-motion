# KITTI真实图像输入与第三阶段时间竞争验证

本轮新增KITTI01和08各80帧，复用已有KITTI04回归片段。选段依据预先保存的GT筛选结果，模型参数在本轮响应产生前固定。

## 输入与时间

| case | 原序列帧号，首尾包含 | 图像尺寸 | 仿真帧间隔 | 本轮用途 |
|---|---|---|---|---|
| kitti01 | 000665–000744 | 160×48 | 104 ms | 新高速道路场景 |
| kitti08 | 002681–002760 | 160×48 | 104 ms | 新街道与复杂纹理场景 |
| kitti04 | 000153–000232 | 160×48 | 104 ms | 既有回归对照 |

新增刺激包位于本机 `D:\A_Gorger\program\MSTD\code\input\stimuli\natural\kitti`。完整刺激ID如下。

```text
kitti01_f000665_f000744__160x48__f000080__dt104ms__gray_resize__v01
kitti08_f002681_f002760__160x48__f000080__dt104ms__gray_resize__v01
kitti04_f000153_f000232__160x48__f000080__dt104ms__gray_resize__v01
```

每个目录包含同名DAT、`.dat.json`、相机内参、帧映射、逐帧GT、artifact及MP4。新增01/08还包含选中帧的相机位姿和逐文件哈希校验记录。

DAT采用项目原生VisualStimulus v1灰度格式，25字节头，逐帧TYX uint8像素。80帧DAT各614425字节。图像使用Pillow LANCZOS缩放，内参采用像素中心映射同步变换。MP4采用H.264、yuv420p，以最近邻放大到1280×384供观看，共80帧、8.32秒。

模型采用固定104 ms时间网格，最近邻曝光映射保留全部连续80帧。01最大曝光偏差27.74 ms，08最大12.80 ms。原始曝光时间与固定仿真时间都写入frame_map.csv。视频8.32秒包含末帧104 ms展示时间，源图像首末曝光跨度在metadata中单独保存。

新增GT包含约1秒中心位移参考、约0.2秒中心参考、当前到下一帧的区间平移方向。expected_heading_deg继续使用既有约1秒定义，方便历史比较。时间跟踪分析需要同时查看更短窗口和因果模型延迟。中心窗口使用的未来位姿仅供离线评价。

## 上传到服务器

复制01和08两个完整刺激目录，保留其目录名与内部文件。已有04目录也需要在服务器上可用，才可以执行可选回归命令。

服务器对应父目录为 `/data/workspace/WYQ/mstd/code/input/stimuli/natural/kitti`。

将本轮新增的 `tools/run_kitti_temporal_inputs.py` 同步到服务器。该脚本调用既有 `tools/run_mstd_temporal_end_to_end.py`，并使用第三阶段已有模型源码和配置。制作刺激的两个新增工具可随源码同步，用于复现与数据检查。

服务器需要已有mstd环境、NumPy和支持CUDA的PyTorch。运行已有刺激时，ffmpeg和Pillow属于本机制作/预览依赖。实际运行还需要之前已同步的第三阶段源码、`configs/model_python_spiking_temporal_baseline32_v01.json` 和投影配置。

服务器上的输入索引由执行工具检查并按缺失项追加，保留服务器已有记录。--plan-only执行只读检查。

## 服务器执行

先在服务器code目录检查数据、完整文件哈希、内参、真值与实际配置。

```bash
cd /data/workspace/WYQ/mstd/code
conda activate mstd

python -u tools/run_kitti_temporal_inputs.py \
  --cases kitti01 kitti08 \
  --backend torch --device cuda --plan-only
```

计划应显示2个case、每个12次运行、合计24次，80帧、104 ms、seed43、substeps32。然后执行完整响应。

```bash
python -u tools/run_kitti_temporal_inputs.py \
  --cases kitti01 kitti08 \
  --backend torch --device cuda
```

可选的KITTI04回归另行执行。

```bash
python -u tools/run_kitti_temporal_inputs.py \
  --cases kitti04 \
  --backend torch --device cuda
```

需要先检查服务器入口时，可使用2帧短跑。其结果用于检查执行连通性。

```bash
python -u tools/run_kitti_temporal_inputs.py \
  --cases kitti01 --max-frames 2 \
  --backend torch --device cuda
```

执行工具遇到失败会停止，终端保留具体异常。修复后可只选失败的case重跑，生成独立时间编号。

## 输出与返回文件

沿用既有端到端入口的输出命名，每个case在服务器 `/data/workspace/WYQ/mstd/code/experiments/validation` 下生成一个父目录。

```text
<北京时间>__mstd_temporal_image__py_spike_cuda__s0043__r01/
  artifact.json
  plan.json
  summary.json
  baseline/
    reference/
    input_only/
    no_self/
    no_lateral/
    tau40/
    tau160/
  spatial_only/
    reference/
    input_only/
    no_self/
    no_lateral/
    tau40/
    tau160/
```

终端每次打印OUTPUT路径，plan.json中的stimulus字段标明对应01、08或04。每个case包含两种冻结MT分支与六种MSTd动力学设置。baseline/reference计算V1，后续各设置重放同一V1，并校验同一MT分支的活动哈希一致。

将每个完整父目录复制回本机 `D:\A_Gorger\program\MSTD\code\experiments\validation`。其中包含逐帧estimates.jsonl、模型配置、运行报告和原生监视器。输入真值保存在刺激目录。完整传回后可进行GT与MSTd曲线、MT方向叠加和跨分支比较。

本轮运行入口输出原生响应。GT误差、有效率和视频比较在返回数据分析时生成。

## 评价内容

- 暖启动剔除前20帧，保留全帧结果供查阅。
- 同时报告未门控候选误差、有效输出误差与有效率。
- 同时报告恒输出0°基线，各片段使用同一参数。
- 检查MT方向结构与空间支持，再解释MSTd的偏向。
- 比较reference、input_only、no_self、no_lateral以及两个时间常数。
- 联合检查稳态偏差、抖动与真实航向变化的延迟，保留有效性拒绝区间。
- 01与08均接近正前方，这轮评价覆盖场景变化，进一步的非零航向覆盖仍需补充。

## Middlebury本轮检查

本机六个下载目录均已检查。24个序列目录对应12组other与12组eval，公开flow10.flo共8组。6组具有8帧图像，Dimetrodon与Venus各2帧。allframes和twoframes中的frame10/frame11逐文件哈希一致。

优先MT候选为RubberWhale、Hydrangea、Urban2，后续可加入Grove2、Grove3、Urban3。flow10表示frame10到frame11的二维位移，单位是源图像像素/帧对。other-gt-interp提供中间图像真值。

当前V1保留9帧历史，需要明确短序列的时间边界、输出延迟与flow10对齐。重复预热帧会引入边界假设，生成协议需要显式记录这一点。源档案没有提供可直接采用的统一物理帧间隔。MT可比较二维方向误差与响应覆盖率，MSTd航向评价还需相机运动真值。

本轮完成Middlebury源数据与FLO完整性检查，MT专项刺激及GPU测试将在短序列边界协议确定后执行。

[Middlebury官方数据说明](https://vision.middlebury.edu/flow/data/)
