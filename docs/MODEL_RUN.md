# 一个项目、一个环境、一次运行

2026-11-08更新：不指定config时统一入口默认Python脉冲MT和GPU，`--device cpu`切回CPU，见[GPU运行说明](GPU_EXECUTION.md)。本文显式使用`configs/model.json`的例子仍为旧连续率基线。

> 2026-11-06：新增Python脉冲MT主线，选择 `--config configs/model_python_spiking.json`，不需要CARLsim/CMake。上传、验证、细时间步反馈及记录格式见[Python脉冲运行说明](PYTHON_SPIKING_RUN.md)。本文原配置仍对应连续率基线。

V1、MT、MSTd共用现有项目及虚拟环境。没有“V1专用虚拟环境”的要求。原`mstd.v1_cli`保留为数值对照工具，日常运行使用统一入口。

## 1. 直接上传文件

本机根目录：`D:/A_Gorger/program/MSTD/code/`。保持相对路径，同步到现有服务器项目根目录，例如`/data/workspace/wyq/mstd/code/`。

必需：

| 路径 | 内容 |
| --- | --- |
| `run_model.py` | 统一运行入口 |
| `pyproject.toml` | 同一环境下的依赖与安装设置 |
| `src/mstd/`整个目录 | 原模型及新V1、连接、配置、监视器；同步整个目录可避免漏传依赖 |
| `configs/model.json` | 计算开关、监视器及运行配置 |
| `configs/v1_projection.json` | 已有投影系数，保持不变 |

验证/查看结果时额外同步：

- `tests/`整个目录。
- `examples/plot_model_monitors.py`、`examples/plot_python_v1.py`、`examples/validate_python_v1_suite.py`。
- 原CUDA数值验收时，再同步`examples/build_cpp_v1_reference.py`、`examples/export_v1_continuous.cpp`及`reference_v1/`整个目录。
- `README.md`、`THIRD_PARTY_NOTICES.md`和相关`docs/`说明。

无需上传`outputs/`、`tmp/`、`.git/`、虚拟环境或压缩包。已有服务器刺激可直接使用原路径。

本轮统一入口新增/修改的源文件为：`model.py`、`monitors.py`、`simulation.py`、`cli.py`、`tensor_pipeline.py`、`__init__.py`，以及根入口、配置和依赖文件。若上一轮V1代码尚未上传，应按上表同步整个`src/mstd/`。

## 2. 现有环境直接运行

激活你现在的项目环境，在服务器项目根目录执行：

```bash
python -m pip install -e .
python run_model.py --config configs/model.json \
  --stimulus /实际路径/刺激.dat \
  --output experiments/scratch/manual__model_run_01__r01 \
  --backend torch --device cuda
```

已有可用NumPy/PyTorch时，无需新建环境或重复安装PyTorch。CPU运行去掉最后一行，默认使用NumPy/CPU。

也可以使用同一项目的模块入口：

```bash
python -m mstd simulate --config configs/model.json --stimulus /实际路径/刺激.dat --output experiments/scratch/manual__model_run_02__r01
```

读取输入后，各层直接通过内存数组传递响应，无需先把V1导成文件再运行MT。监视器是旁路记录；未来实时源也可调用`VisionModel.step(frame,timestamp)`。

## 3. 计算开关与监视器开关

`configs/model.json`中的两个配置块分别控制：

```json
{
  "layers": {"v1": true, "mt": true, "mstd": true},
  "monitors": {
    "v1": true,
    "mt_projection": true,
    "mt_pooling": true,
    "mt_normalization": true,
    "mstd": true,
    "summary": true,
    "every_n_frames": 1
  }
}
```

- `layers`控制实际计算。例如只看V1，可设MT和MSTd为false；只运行V1→MT，可关闭MSTd。
- `monitors`控制哪些响应写到磁盘。关闭V1监视器，V1仍正常为MT提供输入；关闭所有监视器仍会完整计算并生成最终运行报告。
- `every_n_frames=5`表示每5帧保存一次，模型仍处理全部帧；不会跳过滤波器历史或MSTd状态更新。
- 关闭MT时必须同时关闭MSTd。图像输入不能关闭V1后直接送MT，程序会拒绝无效配置。
- 若确实要关闭V1计算，可把输入改为已有的逐帧V1连续记录，并设`layers.v1=false`。稀疏监视记录不能用于宣称逐帧等价回放。

当前MT的“投影、池化、归一化”是可以分别观察的计算阶段，并非本轮新增了三个独立生物脑区。监视的数据是连续响应率/活动，不是膜电位或放电事件。

## 4. 一次运行后的输出

默认全部层和监视器开启：

```text
experiments/scratch/manual__model_run_01__r01/
  model_config.json
  model_report.json
  frames.csv
  estimates.jsonl
  monitors/
    v1/                  # rates.f32 + metadata.json；[T,3,28,H,W]，Hz
    mt_projection/       # values.f32 + metadata.json；[T,8,H,W]，Hz
    mt_pooling/          # values.f32 + metadata.json；[T,8,H,W]，Hz
    mt_normalization/    # values.f32 + metadata.json；[T,8,H,W]，无量纲
    mstd_activity/       # values.f32 + metadata.json；[T,K]，无量纲
    mstd_scores/         # values.f32 + metadata.json；[T,K]，模板匹配分数
```

每个监视器元数据包含shape、单位、采样的原始帧编号及时间戳；MT包含方向角，MSTd包含模板中心位置。输出目录必须尚不存在，避免覆盖实验。

查看结果：

```bash
python -m pip install matplotlib  # 仅缺少绘图库时需要
python examples/plot_model_monitors.py experiments/scratch/manual__model_run_01__r01 \
  --frame 20 --output experiments/scratch/manual__model_run_01_plots__r01
```

`--frame`是原输入帧号；启用稀疏记录后必须选择实际保存的帧。脚本绘制V1通道均值、MT三个阶段空间响应及MSTd活动/分数时间曲线。

## 5. 没有相机内参时，也能观察MSTd

默认`intrinsics=null`时，MSTd使用横向均匀分布的像素坐标模板中心，记录活动与胜出中心，角度读出为空，`bearing_valid=false`。这满足观察神经链路的需要，不虚构相机标定。

有真实或合成刺激配套内参时：

```bash
python run_model.py --stimulus /实际路径/cloud_forward.dat \
  --intrinsics /实际路径/intrinsics.json --output experiments/scratch/manual__selfmotion_model_01__r01 \
  --backend torch --device cuda
```

内参文件使用`fx_px, fy_px, cx_px, cy_px`四个字段。此时模板网格按原-20°至+20°航向范围构造；`mstd_template_count`默认9。无内参的像素模板网格与该角度模板网格不同，不能当作同一个标定条件比较航向精度。

## 6. 怎样验证重构

1. 在同一环境运行`python -m unittest discover -s tests -v`。
2. 用相同配置及刺激，分别运行NumPy/CPU和PyTorch/CUDA，开启V1监视器。
3. 将两个输出的`monitors/v1/`传给`python -m mstd.v1_cli compare`，比较84通道以及后续MT阶段。详细参数和原CUDA连续响应导出步骤见[PYTHON_V1_SERVER.md](PYTHON_V1_SERVER.md)。
4. 只有像素/方向前端通过后，再用相机内参和运动真值验证MSTd航向。

统一入口保留前一轮算法和已知限制：光栅方向检查通过，随机纹理的最强单通道判据未通过，PyTorch/CUDA及原CUDA逐值对照仍待服务器执行。统一项目与监视器的调整不会把这些项目自动标记为通过。

## 7. 2026-11-01服务器报错修复

`uint8 picture DAT (type=4) required`来自旧版本对刺激类别的错误限制。原CARLsim的`stimType_t`定义0=光栅、1=交叉光栅、2=点、3=条、4=图片、5=电影、6=复合刺激，`readFrameChar`对这些类别都读取uint8帧。当前Python读取器和独立C++导出器均接受0..6，仍检查签名、版本、通道、尺寸和精确文件长度。无需修改原DAT头或重新生成刺激。

`environment`输出`"torch": null`表示当前环境没有PyTorch；`pip install -e .`只安装基础NumPy依赖。CPU可立即使用；GPU需要在同一环境补充CUDA版PyTorch。已核对[官方历史版本说明](https://docs.pytorch.org/get-started/previous-versions/)中的2.6.0/cu124构建。以下显式固定该构建，并保留PyPI用于获取其依赖：

```bash
python -m pip install 'torch==2.6.0+cu124' \
  --index-url https://download.pytorch.org/whl/cu124 \
  --extra-index-url https://pypi.org/simple
python -m mstd.v1_cli environment
```

确认`cuda_available=true`后再运行GPU命令。如果安装或导入失败，保留原始错误，不将CPU回退当成GPU通过。

`fatal error: filesystem`来自独立C++导出器对C++17标准库的依赖，尚未进入真实CUDA数值验证。当前导出器改用C++11、标准文件流及原子目录创建，构建脚本/测试同步更改，无需为此先升级服务器全局编译器。

本次运行修复需同步`src/mstd/simulation.py`和`src/mstd/v1_recording.py`。C++兼容修复同步`examples/export_v1_continuous.cpp`和`examples/build_cpp_v1_reference.py`。回归测试同步`tests/test_model_simulation.py`和`tests/test_cpp_v1_exporter.py`。

建议先用原刺激运行20帧CPU预览（完整时间/方向验收仍需整段和预热）：

```bash
python run_model.py --config configs/model.json \
  --stimulus trail/8directions/V1input/selfmotion_8directionsx3speed.dat \
  --output experiments/scratch/manual__model_cpu_preview_01__r01 --max-frames 20 \
  --backend numpy --device cpu
```

该DAT头没有帧间隔字段。默认50ms必须与原CARLsim实验一致；若原实验不同，修改`configs/model.json`的`frame_ms`，不能从文件名推断帧率。

## 8. 完整4800帧：输入刺激与MT响应视频

`plot_model_monitors.py --frame 19`只导出一帧的静态图。逐帧视频使用现有`compare_mt_stages.py`新增的`--recording`模式，直接读取统一模型保存的三个MT监视器，不依赖CARLsim的`spk_V1.dat`，不重新计算MT，也不需要PyTorch或Matplotlib。运行和视频导出均可在现有环境使用NumPy/CPU完成；视频编码需要ffmpeg。

本次同步：

- `examples/compare_mt_stages.py`（更新）。
- `examples/render_model_mt.py`（新增）。
- `configs/model_mt_video.json`（新增）。
- `tests/test_model_mt_video.py`（新增，仅用于回归验证）。
- 依赖原有`examples/visualize_mt_gratings.py`，保持该文件也在服务器相同目录。

在项目根目录执行，输出目录须尚不存在：

```bash
conda activate mstd
cd /data/workspace/wyq/mstd/code
export PYTHONPATH="$PWD/src"
ffmpeg -version

python run_model.py --config configs/model_mt_video.json \
  --stimulus trail/8directions/V1input/selfmotion_8directionsx3speed.dat \
  --output experiments/scratch/manual__python_mt_4800_01__r01 \
  --max-frames 4800 --backend numpy --device cpu

python examples/compare_mt_stages.py \
  --recording experiments/scratch/manual__python_mt_4800_01__r01 \
  --stimulus trail/8directions/V1input/selfmotion_8directionsx3speed.dat \
  --max-frames 4800 \
  --rate-max 300 --activity-max 1 --colormap jet \
  --progress-every 20 \
  --output experiments/scratch/manual__python_mt_video_4800_01__r01
```

该配置默认50ms/帧，须与原实验一致；V1和MT持续计算，MSTd暂关闭，只保存MT三个阶段及摘要，不额外写出大体积84通道V1记录。若已有完整运行且保存了三个MT监视器，无需重跑模型，直接将`--recording`指向已有目录。只有20帧的预览目录无法导出4800帧；稀疏监视器也不能冒充逐帧记录。

得到`experiments/scratch/manual__python_mt_video_4800_01__r01/mt_stages.mp4`：上方为输入图像、帧号和时间，下方三排分别为MT投影率Hz、局部池化输入率Hz、归一化活动（非Hz），每排八个方向，顺序为0°右、45°右下、90°下、135°左下、180°左、225°左上、270°上、315°右上。整段固定色标；大于色标上限的值仅在显示时截顶，原始数组和CSV不变。青色线标记空间平均响应最强的方向。

同时保存`stage_statistics.csv`（逐帧每阶段的八方向均值、峰值方向和色标饱和比例）、`video_config.json`（帧数、时间、输入SHA256、方向顺序及配对信息）及第一/末帧PNG。默认`--playback-rate 1`，50ms/帧对应20FPS，4800帧对应240秒。帧i与模型处理帧i后的输出配对，不移除启动历史或自动移动时间轴。

无需manifest也可查看真实刺激和响应；这时画面不虚构方向/速度标签。若已有确实对应本DAT的序列manifest，可在第二条Python命令增加：

```bash
--manifest input/stimuli/grating/grating_sequence__80x80__f004800__dt50ms__v02/manifest.json
```

程序先核对DAT哈希、尺寸和时间间隔，再读取分段标签供视频显示。标签不参与神经模型计算。旧`--spikes`/`--cache`模式保持原用法。

## 9. 真实图片文件夹

新增`examples/images_to_dat.py`按自然数字顺序把图片文件夹转为灰度/RGB DAT，支持保留原尺寸、显式缩放和居中补边。daria_walk的180×144及188×144样例、上传与GPU视频命令见[IMAGE_SEQUENCE_DAT.md](IMAGE_SEQUENCE_DAT.md)。模型从DAT自动读取宽高。
