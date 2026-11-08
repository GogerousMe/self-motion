# Python V1 v0.5：上传、运行和验收

日期：2026-10-28。目标服务器为用户提供的 RTX A5000 24GB；先离线处理，保留逐帧接口。所有命令在现有项目根目录运行；输出目录必须不存在，重复实验请换编号。

V1、MT、MSTd共用同一个项目和环境。日常运行已统一为`python run_model.py ...`，详见 [MODEL_RUN.md](MODEL_RUN.md)。本文后续命令用于前端数值验收，不代表需要单独部署V1项目。

## 1. 上传什么

直接同步代码到现有服务器项目，不要求压缩包。文件清单：

- `src/mstd/`：整个目录，包含统一模型、监视器及数值对照模块。
- `configs/`：现有28×8投影系数，首轮保持不变。
- `examples/`：运行/绘图/方向套件脚本、独立C++连续响应导出器、隔离编译工具。
- `reference_v1/`：用户确认的V1源码及匹配头文件快照；只为独立数值基准使用。
- `tests/`、`docs/`、`README.md`、`pyproject.toml`、`THIRD_PARTY_NOTICES.md`。
- 根目录的`run_model.py`是统一运行入口。

无需上传本机原始大DAT、历史脉冲、所有CARLsim核心、Python临时依赖或测试输出。`reference_v1/`只在运行独立C++对照时需要，其中的源/头文件必须配套。

## 2. 环境与自检

Python>=3.10，激活现有项目环境即可，全部层共用该环境：

```bash
python -m pip install -e '.[plots]'
python -m mstd.v1_cli environment
python -m unittest discover -s tests -v
```

已有可用PyTorch时无需重装。仅在缺少PyTorch且需要GPU时，才在同一环境安装匹配驱动的构建，例如`python -m pip install 'torch==2.6.0+cu124' --index-url https://download.pytorch.org/whl/cu124 --extra-index-url https://pypi.org/simple`。显式固定CUDA构建并保留PyPI用于依赖；版本来源见[官方说明](https://docs.pytorch.org/get-started/previous-versions/)。本项目不需要torchvision/torchaudio。

`environment`应显示CUDA可用和RTX A5000。若CUDA不可用，仍可运行NumPy版，但不要把它当作GPU测试通过。`nvidia-smi`的CUDA字段不代表PyTorch实际使用的CUDA版本。通常无需为PyTorch另行编译CUDA内核；第5节原C++数值基准才需要原有Toolkit和nvcc。

测试中的PyTorch数值对照会运行torch CPU，并在CUDA可用时追加GPU；没有NumPy或PyTorch时会明确skip。C++导出器单元测试使用mock头文件验证C++11编译及捕获文件格式，不能替代第5节真实CUDA比较。2026-11-01已移除`<filesystem>`依赖；DAT读取也已支持原CARLsim的全部0..6类别，见[报错修复说明](MODEL_RUN.md#7-2026-11-01服务器报错修复)。

## 3. 第一轮：同一刺激的CPU/GPU结果

先生成80×80、八方向独立光栅，每条件40帧、50ms/帧，前20帧预热：

```bash
python -m mstd generate-calibration --output input/stimuli/grating/grating_calibration__80x80__f000040__dt50ms__8dirs__v01 \
  --width 80 --height 80 --frames 40 --warmup-frames 20

python -m mstd.v1_cli run input/stimuli/grating/grating_calibration__80x80__f000040__dt50ms__8dirs__v01/grating_d000.dat \
  --output experiments/scratch/manual__python_cpu_d000_01__r01 --projection-config configs/v1_projection.json \
  --backend numpy --device cpu --frame-ms 50 --save-raw --save-mt \
  --warmup-frames 20 --expected-direction-deg 0

python -m mstd.v1_cli run input/stimuli/grating/grating_calibration__80x80__f000040__dt50ms__8dirs__v01/grating_d000.dat \
  --output experiments/scratch/manual__python_gpu_d000_01__r01 --projection-config configs/v1_projection.json \
  --backend torch --device cuda --frame-ms 50 --save-raw --save-mt \
  --warmup-frames 20 --expected-direction-deg 0

python -m mstd.v1_cli compare \
  experiments/scratch/manual__python_cpu_d000_01__r01/raw_v1 experiments/scratch/manual__python_gpu_d000_01__r01/raw_v1 \
  --output experiments/scratch/manual__compare_cpu_gpu_01__r01 --projection-config configs/v1_projection.json
```

查看 `experiments/scratch/manual__compare_cpu_gpu_01__r01/comparison.json`：

- `raw_passed`：全部84通道、所有像素、所有帧满足 `abs(candidate-reference) <= atol + rtol*abs(reference)`。
- `stage_passed`：八方向投影、局部池化、MT归一化三个阶段分别满足容差。
- `passed`：以上均通过。失败时CLI退出码为2，并仍保留报告。
- `channels.csv`给出每个scale/filter的RMSE；`frames.csv`给出逐帧原始和各MT阶段误差。

默认`atol=rtol=5e-4`是初始浮点工程容差，不是生理标准。不能为获得通过而直接放宽容差；先看量级、通道、增益、坐标和源版本。比较严格要求输入SHA256、源配置SHA256、尺寸、帧数及时间间隔匹配，不做自动增益拟合或时间平移。

本轮只证明两个Python计算后端一致，还需要第5节独立C++基准。

## 4. 第二轮：八方向、空间响应及时间响应

```bash
python examples/validate_python_v1_suite.py input/stimuli/grating/grating_calibration__80x80__f000040__dt50ms__8dirs__v01/manifest.json \
  --output experiments/validation/manual__v1_grating_suite_gpu_01__r01 --projection-config configs/v1_projection.json \
  --backend torch --device cuda --save-mt

python examples/plot_python_v1.py experiments/scratch/manual__python_gpu_d000_01__r01 \
  --frame 30 --output experiments/scratch/manual__python_gpu_d000_plots_01__r01
```

`directions.csv`和`suite_report.json`报告八条件的MT胜出方向、群体方向和集中度。初始工程判据：预热后有响应、集中度>=0.1、最强通道误差<=22.5°。期望方向只用于事后评估，不参与模型计算。

绘图输出包括：

- `time_courses.png`：八方向投影率、MT活动和证据的时间曲线。
- `direction_rates.png`：同帧八方向投影空间图，单位Hz。
- `mt_activity.png`：同帧八方向归一化空间图，无量纲，需`--save-mt`。
- `v1_channel_means.png`：84通道ROI均值，需`--save-raw`。

每一组空间图内部共用色标；Hz与无量纲图不能仅凭颜色比较幅值。检查是否翻转、始终偏向同一方向、空间结构异常，以及启动过程。当前9帧历史滤波带来的延迟不自动移除；保持相同帧序号对照。

光栅之后，用随机纹理复核：

**本机已观察到：以下固定随机纹理条件下，八个条件的峰值通道均偏到相邻45°，因此套件应报告未通过；群体向量方向误差约0.17°–4.13°。保留这个失败，不更改矩阵或按标签选择通道。** 这是扩大到纹理后的模型响应诊断，不等价于数值移植错误；先按第5节与原CUDA响应对照定位。

```bash
python -m mstd generate-calibration --output input/stimuli/texture/texture_calibration__80x80__f000040__dt50ms__8dirs__v01 \
  --width 80 --height 80 --frames 40 --warmup-frames 20 --kind texture
python examples/validate_python_v1_suite.py input/stimuli/texture/texture_calibration__80x80__f000040__dt50ms__8dirs__v01/manifest.json \
  --output experiments/validation/manual__v1_texture_suite_gpu_01__r01 --projection-config configs/v1_projection.json \
  --backend torch --device cuda
```

再加入静态纹理、黑场、方向切换和多速度；连续切换内部保留历史，独立序列才reset。八方向通过不能证明孔径问题已解决，也不能作为MSTd自运动精度验收。

## 5. 第三轮：与原始CUDA V1的连续响应比较

这一轮不需要构建CARLsim网络。工具直接调用MotionEnergy的帧入口，在泊松采样之前保存84通道浮点响应。

### 5.1 隔离编译固定源码

项目`reference_v1/`包含用户确认源码的快照，原工程不会被修改。下例使用原CUDA11.4；如果实际Toolkit在其他位置，修改`--cuda-root`：

```bash
python examples/build_cpp_v1_reference.py \
  --cuda-root /usr/local/cuda-11.4 --arch sm_86 \
  --output build/cpp_v1_build_01
```

若找不到`helper_cuda.h`，加上 `--cuda-samples-include /实际包含helper_cuda.h的目录`，可沿用原工程使用的CUDA samples include路径。该头文件属于CUDA样例辅助库，不属于本项目。

编译工具检查源码SHA256，并生成 `build.json`，记录编译命令、nvcc版本及源、头、库哈希。原有动态CARLsim库无需链接。此步骤需要Linux/CUDA实际执行，本机Windows回归不声称已完成该GPU编译。

### 5.2 同一输入、同样帧数捕获

```bash
build/cpp_v1_build_01/export_v1_continuous \
  input/stimuli/grating/grating_calibration__80x80__f000040__dt50ms__8dirs__v01/grating_d000.dat experiments/scratch/manual__cpp_v1_d000_01__r01 50

python -m mstd.v1_cli register-cpp experiments/scratch/manual__cpp_v1_d000_01__r01 \
  --stimulus input/stimuli/grating/grating_calibration__80x80__f000040__dt50ms__8dirs__v01/grating_d000.dat \
  --source reference_v1/src/motion_energy.cu \
  --library build/cpp_v1_build_01/libME_reference.a

python -m mstd.v1_cli compare \
  experiments/scratch/manual__cpp_v1_d000_01__r01 experiments/scratch/manual__python_cpu_d000_01__r01/raw_v1 \
  --output experiments/scratch/manual__compare_cpp_python_01__r01 --projection-config configs/v1_projection.json
```

先右方向通过，再扩大到其余方向、随机纹理、非正方形/RGB、启动与黑场。若用导出器最后一个可选参数限制帧数，例如10帧，则Python也加`--max-frames 10`，并更换输出目录。短预览不代表整段验证。

该比较验证Python与提供的固定CUDA源码一致。若需证明与历史成功实验的二进制库一致，还应对当时实际库做同样导出，核对其编译来源；目录内源码相同不自动证明安装库相同。

### 5.3 不通过时返回什么

返回 `comparison.json`、`channels.csv`、`frames.csv` 和 `cpp_v1_build_01/build.json`，以及失败条件名。无需先返回整个大响应文件。不要补零、重排通道、翻转图像、调整增益或移动时间序列来强行匹配。

`spk_V1.dat`是随机脉冲实现，不应与连续响应逐值相减。当前版本输出连续率，尚未实现脉冲仿真兼容模式；历史脉冲仍可走原有`convert-spikes`流程作统计和任务层面的对照。

## 6. MT如何连接，MSTd如何继续运行

`run`已经自动执行以下步骤，无需再手工构建神经元间连接：

```text
图像DAT -> Python V1 [3,28,H,W] 连续率
        -> 原投影系数、尺度汇总、有符号求和后整流 [8,H,W]
        -> 原MT局部平均与方向归一化 [8,H,W]
        -> 可选MSTd径向模板及原参考竞争方程
```

数组后端实现与现有标准库`project_counts()`、`encode_mt()`和`MSTd.step()`已有数值回归测试，旧文件保持不变。连续率投影不会重复乘以`1000/bin_ms`，也不会重复做方向投影。

没有相机内参时只跑到MT，避免用虚构内参输出航向。使用已有自运动刺激及其对应的真实/合成内参：

```bash
python -m mstd generate-selfmotion --output input/stimuli/selfmotion/selfmotion_suite__80x80__f000040__dt50ms__v01 --frames 40

python -m mstd.v1_cli run input/stimuli/selfmotion/selfmotion_suite__80x80__f000040__dt50ms__v01/cloud_forward.dat \
  --output experiments/scratch/manual__python_selfmotion_forward_01__r01 --projection-config configs/v1_projection.json \
  --backend torch --device cuda --frame-ms 50 \
  --intrinsics input/stimuli/selfmotion/selfmotion_suite__80x80__f000040__dt50ms__v01/intrinsics.json
```

输出`estimates.jsonl`为MSTd逐帧结果；输入/估计时刻分别标记。该目录也包含标准 `metadata.json + rates.f32`，可直接交给旧CLI再次运行：

```bash
python -m mstd run experiments/scratch/manual__python_selfmotion_forward_01__r01 \
  --output experiments/scratch/manual__selfmotion_reference_replay_01__r01 --last-frame-duration-s 0.05
```

旧CLI重放用于接口与后端方程对照，不能替代自运动精度实验。航向验收使用生成器的`.truth.jsonl`，优先测试静止、前进和相机朝向固定的左右斜向平移；光栅平移方向不等于MSTd自运动航向。

## 7. 大小、存储和实时接口

先80×80/108×108，再逐级测试更大尺寸。`run_report.json`记录总耗时、V1耗时、后端耗时和墙钟FPS；含诊断与文件写入，不是纯GPU核吞吐。GPU计时已经同步。

80×80×84×40×4字节约82MiB；一个CPU和一个GPU原始记录合计约164MiB。长序列默认不保存84通道原始响应，需要数值对照时才加`--save-raw`。图像不自动缩放，改变尺寸或裁剪时必须同步内参与感受野解释。

在线预留接口是：

```python
model = MotionEnergyV1(width, height, frame_interval_s=0.05, backend='torch', device='cuda')
model.reset()  # 仅独立序列边界
rates = model.step(uint8_frame, timestamp_s)
```

返回所选设备上的SFYX数组。当前严格要求固定输入帧间隔；未来实时视频若丢帧，需要明确重采样/状态策略，不能直接忽略时间间隔。

## 8. 当前验证边界

本机已通过NumPy数学/状态/协议及原后端对照测试，并运行80×80八方向光栅，八个MT胜出通道均正确。随机纹理峰值判据未通过，见上文；108×108右移光栅也通过。实际CUDA库数值对照、服务器GPU速度和真实视频性能仍需按本文在服务器运行；详细记录见 [PYTHON_V1_VALIDATION.md](PYTHON_V1_VALIDATION.md)。
