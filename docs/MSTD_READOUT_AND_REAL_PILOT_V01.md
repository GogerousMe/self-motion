# 连续航向读出与本地真实图像试运行

2026-11-09。继续MSTd自运动航向主线：增加独立读出候选，保留离散赢家，并开始近直行真实图像探索。当前结果支持“受控、近纯平移场景中的部分水平航向估计能力”，尚不是所有简单运动均通过、真实视频泛化或完整位姿估计的证明。

## 1 连续读出已经接入，默认仍为winner

配置字段：

```json
"mstd_readout": {"mode": "local_quadratic", "curvature_epsilon": 1e-12}
```

默认mode为winner；候选配置为 `configs/model_python_spiking_geometry_quadratic_v01.json`。MSTd模板、竞争状态和几何有效性判据保持原机制。仅在标定完整、有效输入且候选为均匀升序水平角度网格时，对峰及其两个邻居作局部抛物线角度读出；边界、退化或不支持的网格回退赢家。

运行时同时保存：

- alpha_x_deg：所选模式的有效航向。
- winner_alpha_x_deg：离散赢家航向。
- local_quadratic_alpha_x_deg：候选航向，发生回退时等于赢家。
- readout：方法、回退原因和局部偏移。

无效输入或没有内参时，对外航向均为null。没有把像素间距误当等角度间距，也没有改变原生脉冲或竞争活动。新增检查脚本会分别报告赢家和所选读出的无条件误差，避免新读出仍被误当成赢家评价。

正式函数已经对13段保存状态、1040帧完成离线回归，与此前探索性拟合结果一致。结果位于：

```text
experiments/validation/20261109__mstd_readout_recordings__py_rate_cpu__r01/
```

已知镜像组平均MAE由2.358°变为1.194°；旧模板中心实验存在回退，例如地面0°由1.000°变为1.529°。这属于已知数据上的实现回归；候选尚未通过新的独立验证，旧实验原判定未改写。

## 2 本地数据实际有哪些

原始目录只读：

```text
E:\SLAM数据\KITTI
E:\SLAM数据\TUM
```

KITTI的灰度image_0及00–10位姿、calib.txt、times.txt均可用；11个序列的图像数、位姿数、时间戳数一致。11–21属于官方无公开真值的测试部分，本轮没有用它们做精度筛选。

TUM的RGBD目录已有解压的freiburg1_desk，以及fr1_xyz、fr2_xyz等压缩包。已直接读取两个xyz包的RGB时间戳和位姿元数据，未解压或修改原数据。fr1_xyz有798张RGB，fr2_xyz有3669张RGB；两者均包含前后、侧向及其他运动，不能将整段当成当前扩张模型的正例。

筛选报告：

```text
experiments/validation/20261109__real_sources_screening__py_rate_cpu__r01/
```

筛选只用GT，不看模型结果：80帧中至少95%满足向前速度、水平航向±15°、垂直方向±5°、约1s旋转率≤1°/s与逐步旋转率≤3°/s。KITTI发现12个非重叠候选窗口；上述两个TUM包没有窗口同时满足本组约束。这不意味着TUM不能使用，只表示需另选长度/运动范围，或在后续支持更多运动时测试。

GT平移航向由camera-to-world位姿计算：

```text
v_camera = R_i.T * (p_hi-p_lo) / dt
heading_x = atan2(v_camera.x, v_camera.z)
```

它不是世界坐标轨迹转角，也不是相机yaw。KITTI使用i±5帧约1s的平移方向参考以降低短位移差分的不稳定；该未来GT只用于离线评估，不输入模型。TUM仅用最近位姿匹配作候选筛选，正式评测仍需单独处理时间关联与畸变。

## 3 已准备KITTI04真实图像pilot

选择04序列原始第153–232帧，共80帧。GT参考平均水平航向约−0.128°，范围−0.327°～+0.026°；平均速度14.87m/s，逐步旋转率95分位约1.53°/s，最大约2.00°/s；最大垂直方向约1.71°。这是近直行而非严格零旋转。

准备目录：

```text
input/stimuli/natural/kitti/kitti04_f000153_f000232__160x48__f000080__dt104ms__gray_resize__v01/
```

包含DAT、同名.dat.json、intrinsics.json、逐帧真值、frame_map.csv、artifact.json及preview/input.mp4。只输入灰度图像，不使用Velodyne、深度或GT作为模型驱动。

原图1226×370，缩放至160×48，全视野无裁剪/补边；记录精确x/y比例，并按Pillow像素中心映射更新内参：

```text
fx' = fx*sx; fy' = fy*sy
cx' = (cx+0.5)*sx-0.5; cy' = (cy+0.5)*sy-0.5
```

得到fx=92.27944、fy=91.73075、cx=78.11498、cy=23.31973。不要套用此前80×80、焦距40的内参。

此段实际帧间隔约104ms。为适配当前固定步长模型，使用104ms目标时间网格作显式最近帧关联，保留原80帧无丢弃/重复，最大时间偏移约5.02ms；原始曝光时间逐帧保留。预览H264、125/13fps、80帧、8.32秒。不能把它当成50ms刺激输入。

当前片段接近0°，只能检验真实图像中的近前进可用性，不能证明任意航向或转弯能力。中帧以道路、树木等背景为主，前方也有车辆；动态物体、光照、残余旋转和滤波延迟都是试运行的限制。后续需要匹配的静态冻结画面、更多方向和独立片段。

## 4 同步哪些文件

继续直接复制源码和自包含刺激包，无需打zip。新增/更新：

```text
src/mstd/readout.py
src/mstd/model.py
src/mstd/mstd.py
src/mstd/tensor_pipeline.py
src/mstd/egomotion_truth.py
examples/check_mstd_validity.py
examples/check_mstd_readout_recordings.py
configs/model_python_spiking_geometry_quadratic_v01.json
tools/run_real_heading_pilot.py
tools/prepare_kitti04_heading_pilot.py
tools/screen_real_heading_sources.py
tests/test_mstd_readout.py
tests/test_egomotion_truth.py
docs/MSTD_READOUT_AND_REAL_PILOT_V01.md
input/stimuli/natural/kitti/kitti04_f000153_f000232__160x48__f000080__dt104ms__gray_resize__v01/
```

服务器只需复制这个小刺激包，不需要复制整个E盘数据集；原始外部路径仅用于溯源。运行脚本会校验并追加缺少的刺激登记，保留服务器已有索引。

## 5 服务器命令

```bash
conda activate mstd
cd /data/workspace/WYQ/mstd/code
python tools/run_real_heading_pilot.py --plan-only
python tools/run_real_heading_pilot.py --backend torch --device cuda
```

脚本从DAT元数据读取160×48、104ms和真实缩放内参，不需要手工改配置；默认同时保存赢家与local_quadratic读出。运行80帧，使用种子43。模型只读取图像与标定；GT在运行完成后由评估函数读取。

自动生成：

```text
experiments/validation/<time>__kitti04_f000153_f000232_160x48_f000080__py_spike_cuda__s0043__r01/
```

包含原生V1/MT/MSTd响应、配置、estimates.jsonl、artifact.json，以及heading_metrics.json和HEADING_REPORT.md。报告分别给出两种读出的误差、有效率、95分位和最大误差。

第一轮真实pilot不直接套用合成数据2.5°目标，也不是KITTI官方位姿评测；先判断是否有合理航向、拒绝率是否异常，再定位误差。执行完成不等于真实泛化通过。

## 6 资料依据

- [KITTI官方里程计页面](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)：00–10公开真值、标定和开发包。
- 本机官方开发包KITTI_POSE/devkit/readme.txt：image_0已校正/整流，P0标定、时间戳及相机到第0帧的位姿定义。
- [TUM官方序列说明](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download)：xyz序列用于平移调试，但仍需按运动范围筛选。
- [TUM官方格式与内参](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats)：RGB相机光心位姿、四元数与相机标定。
