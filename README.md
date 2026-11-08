# Python V1–MT–MSTd 自运动模型

**2026-11-08：Python脉冲主线默认GPU。** `run_model.py`默认使用`model_python_spiking.json`与PyTorch/CUDA；加`--device cpu`即可切换NumPy/CPU。环境检查、运行命令和验收指标见[GPU运行说明](docs/GPU_EXECUTION.md)。GPU性能尚待服务器实测。

**文件管理入口：** 所有会话先读 [AGENTS.md](AGENTS.md) 和 [目录与命名规范](docs/FILE_ORGANIZATION.md)。刺激在 `input/`，验证在 `experiments/`，正式响应和展示按运行编号分别存放；旧路径见 [迁移索引](docs/migrations/20261107_layout/paths.csv)。

**2026-11-06：Python脉冲MT主线已接入。** 使用 `configs/model_python_spiking.json`，无需CMake/CARLsim库；见[运行、上传与反馈接口](docs/PYTHON_SPIKING_RUN.md)。MT在Python中维护膜电位、E/I突触和延迟；MSTd暂为逐神经时间步更新的连续适配器。旧 `model_spiking.json` 仅用于CARLsim离线对照。

V1、MT、MSTd 属于同一个模型项目，使用同一个 Python 环境。v0.5.1 提供统一入口 `run_model.py`，一次读取刺激并串联各层，支持计算开关及中间响应监视器，也保留现有 CARLsim 脉冲接口。
当前版本是可运行的纯平移参考原型，不是 Layton 论文的逐式复现，也不是完整自运动或 SLAM 系统。

**日常运行、层开关、监视器及直接上传文件清单见 [统一模型运行说明](docs/MODEL_RUN.md)。不要求单独的V1环境，也不需要压缩包。**
CPU/GPU 和原 C++ V1 数值对照工具见 [Python V1 部署与验收](docs/PYTHON_V1_SERVER.md)。统一模型依赖 NumPy，GPU 使用同一环境中的 PyTorch；原 CUDA 库数值等价性与目标 GPU 性能需在服务器验收。

- [完整技术方案](docs/TECHNICAL_PLAN.md)
- [V1 接口协议与 C++ 接入说明](docs/V1_INTERFACE.md)
- [开发进度与下一步](docs/DEVELOPMENT.md)
- [本机实际验证记录](docs/VALIDATION.md)
- [CARLsim 脉冲 DAT 接入](docs/CARLSIM_IMPORT.md)
- [Linux 方向校准采集流程](docs/CALIBRATION.md)
- [自运动刺激：前进、后退与曲线路径](docs/SELFMOTION_STIMULI.md)
- [自运动估计到 SLAM 位姿估计的分阶段方案](docs/SELF_MOTION_TO_SLAM.md)
- [Ubuntu 20.04 部署与连续八方向光栅 MT 测试](docs/UBUNTU20_MT_GRATING.md)
- [输入光栅与八方向 MT 空间响应同步视频](docs/MT_RESPONSE_VIDEO.md)
- [投影、池化、归一化三阶段 MT 对照验证](docs/MT_STAGE_VALIDATION.md)
- [下一轮合并刺激：纹理对照、连续切换和自运动基线](docs/NEXT_STIMULUS_SEQUENCES.md)
- [测试实验需求与实施方案](docs/EXPERIMENT_PLAN.md)
- [Python V1 迁移评估](docs/PYTHON_V1_FEASIBILITY_REVIEW.md)
- [模型对照图](docs/model-comparison.html)
- [MATLAB 独立方向、静态和多速度光栅](matlab/README.md)
- [MATLAB 连续八方向光栅](srcipts/README.md)

## 快速运行

Python 3.10+，在现有项目环境、项目根目录运行：

```bash
python -m pip install -e .
python -m mstd generate-calibration --output input/stimuli/grating/grating_calibration__80x80__f000040__dt50ms__8dirs__v01 --frames 40 --warmup-frames 20
python run_model.py --config configs/model.json --stimulus input/stimuli/grating/grating_calibration__80x80__f000040__dt50ms__8dirs__v01/grating_d000.dat --output experiments/scratch/manual__grating_80x80_f000040__py_rate_cpu__r01
```

服务器已有可用PyTorch/CUDA时，在最后一条命令追加 `--backend torch --device cuda`。默认监视全部层；开关见 `configs/model.json`。无内参也可记录MSTd图像坐标模板活动，有内参才输出校准航向角。

旧标准库参考后端的接口演示仍可运行（PowerShell）：

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m unittest discover -s tests -v
python -m mstd demo --output experiments/scratch/manual__demo_01__r01
python -m mstd run experiments/scratch/manual__demo_01__r01/recording --output experiments/scratch/manual__replay_01__r01 --last-frame-duration-s 0.05
```

输出路径必须尚不存在，避免覆盖实验。安装环境具备 setuptools 时，也可 `python -m pip install -e .`，随后使用 `mstd` 命令。

演示生成 16×16、8 方向的人工径向响应，经过文件读写、MT、MSTd，输出中心平移方向估计。
这是接口与几何计算的 smoke test；它没有验证真实视频前端、动态物体鲁棒性或曲线运动。

## 当前边界

- 统一入口输入：原始图像刺激DAT；关闭V1时可回放已有84通道连续响应。旧参考入口保留方向率输入。
- 模型：局部池化／归一化、径向膨胀模板、简化分流竞争动力学。
- 输出：投影方位角 alpha_x/alpha_y、活动分布、有效标志和未校准的可靠性指标。
- 当前 CLI 仅扫描水平 -20° 至 +20°，5° 间隔；不输出 yaw、转向状态、曲率或绝对速度。
- v0.2 支持 `inspect-spikes` 与 `convert-spikes`，读取84通道 CARLsim SpikeMonitor DAT，并通过显式28→8投影转成模型输入；用法与限制见上面的接入文档。
- 原始连续响应须使用项目规定的元数据协议，不支持任意无头浮点文件、网络快照或HDF5。
- 没有修改现有 C++ V1 工程；真实导出仍需确认其实际链接库与输出维度。

## 当前开发重点

Python 是模型主体，C++ V1 作为独立数值对照。监视器保存连续率或活动，不冒充膜电位/脉冲仿真；相机内参和自运动真值验证后，才能报告航向精度。
原始DAT、二进制实验输出与本机数据均不上传；测试使用临时生成的小型夹具。

v0.3 新增 `generate-calibration` 与 `analyze-calibration`：生成八方向光栅／纹理刺激，并对真实V1响应输出方向误差及滤波器活动报告。缺少响应时明确标为pending；Linux采集流程见上面的说明。

v0.4 新增 `generate-selfmotion --output input/stimuli/selfmotion/selfmotion_suite__80x80__f000080__dt50ms__14cases__v01 --video`：在静态三维点云／地面点阵中模拟相机运动，导出真实图像AVI、V1输入DAT、相机参数和逐帧真值。包含14个条件；后退／旋转／曲率解码仍需扩展后端。
# CARLsim 脉冲 MT（可选后端）

新增离线 `V1连续响应 → CARLsim泊松输入 → MT兴奋/抑制神经元 → 脉冲率读出 → 连续MSTd`。
实施方案、模型边界、指标及构建命令见 [SPIKING_MT_PLAN.md](docs/SPIKING_MT_PLAN.md)，已执行验证见 [SPIKING_MT_VALIDATION.md](docs/SPIKING_MT_VALIDATION.md)。
使用 `python -m mstd simulate --config configs/model_spiking.json`，先设置真实输入与CARLsim可执行文件路径。
默认回调契约匹配本地修改版内核；链接上游标准CARLsim库时改为 `carlsim.callback_layout=standard`。
旧连续后端默认不变；本版保留旧三群体构网开销，未声称与历史GPU快照等价。
