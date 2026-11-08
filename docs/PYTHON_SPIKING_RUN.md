# Python 脉冲 MT：运行、接口和验证（2026-11-06）

2026-11-08更新：统一入口和本页验证脚本默认PyTorch/CUDA，CPU运行追加`--device cpu`。模型无需CARLsim/CMake。设备规则、服务器命令和GPU验收指标见[GPU_EXECUTION.md](GPU_EXECUTION.md)。本机尚未实测GPU性能。

MT输出解析、方向/时序/脉冲分析及结果登记步骤见[MT响应分析流程](MT_RESPONSE_ANALYSIS.md)。

## 当前实现

主线后端 `mt_backend=python_spiking` 完全在Python内存中运行，不调用CARLsim、不需要CMake或原生动态库。
已有连续后端 `rate` 和可选离线对照后端 `carlsim` 保留。不要把旧 `model_spiking.json` 当成新主线配置；新配置为 `model_python_spiking.json`。

链路：Python MotionEnergy V1 → 本地泊松事件 → 延迟/快速电导突触 → Izhikevich MT E/I → 因果活动迹 → 每神经时间步更新的连续MSTd。
MSTd尚未脉冲化。反馈接口已工作，但默认关闭，没有宣称某种反馈改善自运动估计。

## 同步到服务器

从本地code目录同步以下内容并保持相对结构：

- `src/mstd/` 整个目录（新增neural.py、spiking_mt.py，更新model.py、monitors.py、simulation.py）。
- `configs/model_python_spiking.json`、`configs/v1_projection.json`。
- `examples/validate_python_spiking.py`、`tests/test_python_spiking.py`；完整回归需要同步整个tests目录。
- `run_model.py`、`pyproject.toml`，以及本文和PYTHON_SPIKING_NETWORK_PLAN.md。

这条主线不需上传native、编译产物、旧CARLsim库、outputs或tmp。

## 运行

使用已有Python>=3.10与NumPy环境，在服务器项目根目录：

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m unittest discover -s tests -p test_python_spiking.py -v

# 自动生成受控率输入及32×32右移光栅，并跑完整链路：
python examples/validate_python_spiking.py --output experiments/validation/manual__python_spiking_check_01__r01 --record-spikes

# 扩展为八方向独立光栅：
python examples/validate_python_spiking.py --output experiments/validation/manual__python_spiking_8dirs_01__r01 --all-directions --record-spikes
```

输出目录必须不存在。validation.json中的controls和directions分别记录机制筛查和方向峰值误差；任何失败保留结果并返回非零退出码。
图像条件为32×32、40帧、50ms/帧、1.5px/帧，前20帧预热，ROI排除边缘4px。一个种子/速度上的八方向成功不构成普遍光流或航向精度验证。

实际输入：

```bash
python run_model.py --config configs/model_python_spiking.json \
  --stimulus /absolute/path/to/stimulus.dat \
  --max-frames 40 --output experiments/scratch/manual__python_spiking_real_01__r01
```

图像不会自动缩放。先使用短、小尺寸输入。提供真实对应内参才报告标定航向角；未提供时仍计算MSTd活动。
以上命令默认GPU；CPU参考运行追加`--device cpu`。本机未安装PyTorch，不将GPU支持视为已验收，也不要求随机事件逐个匹配NumPy。

## 状态和数值约定

- 默认神经步长1ms、4个Euler子步（0.25ms）；膜电位v/恢复变量u同步更新，阈值30后复位。数值溢出在复位前拒绝，不用截断掩盖。
- MT E/I均先用a=.02、b=.2、c=-65、d=8；有效电导单位为项目模型单位，尚未标定为生理nS。
- 快速兴奋/抑制电导时常数默认5/10ms，反转电位0/-80mV；不实现慢受体、额外不应期或塑性。
- V1使用每bin精确泊松计数，允许多个事件。计数不裁为0/1；bin内事件聚合到步终点。不是CARLsim不应期采样器。
- 每步先读完全部延迟输入，再更新MT E/I，最后写新事件。delay_ticks=1表示tick k末产生的事件在tick k+1起点到达。E/I计算顺序不改变结果。
- V1正/负投影分别驱动E/I；MT-I输出通过延迟驱动E的抑制电导。固定连接由张量收缩表达，不枚举全局神经元配对。
- frame_ms必须为dt_ms整数倍，mt_radius必须为0。默认不加入空间池化和神经归一化回路。
- 输出滤波率为对bin率进行指数低通，trace_tau_ms默认20；读出活动r/(1Hz+Σr)是适配器，不是电导网络归一化。

## 在线接口与反馈

```python
from mstd.model import VisionModel, load_model_config

config = load_model_config('configs/model_python_spiking.json')
model = VisionModel(32, 32, config)

def controller(ctx):
    # 仅为连通性示例：不宣称此均匀调制具有生物依据或改善航向。
    if ctx['mstd'] is None:
        return None
    drive = 0.5 * max(ctx['mstd']['activity'])
    return {'exc_current': model.ops.zeros((8, 32, 32)) + drive}

model.set_feedback(controller)  # 默认没有控制器
# state = model.step(frame_uint8_HxW, timestamp_s)
# model.set_feedback(None)      # 后续输出停止，已排队反馈仍按延迟到达
# model.reset()                 # 只在独立序列边界清空状态/RNG/队列；保留控制器注册
```

控制器在每个细时间步结束时接收拥有独立副本的context：tick、timestamp_s、MT活动/方向向量、V1/E/I脉冲及当前过渡MSTd状态。
可返回任意子集，数组形状均为[8,H,W]：

| 键 | 作用 |
|---|---|
| exc_current | 加到MT-E的外部有效电流，可正可负 |
| inh_current | 加到MT-I的外部有效电流，可正可负 |
| feedback_exc_events | 非负加权反馈事件，驱动MT-E独立的兴奋电导 |
| feedback_inh_events | 非负加权反馈事件，驱动MT-E独立的抑制电导 |

返回值通过feedback_delay_ticks排队，不能影响控制器已观察的那个时间步。后续MSTd脉冲经过连接权重映射可直接走反馈事件通路。
当前MSTd是连续率适配器，按MT本步因果活动顺序更新；没有冒充多脑区脉冲网络的完全同步更新。
每步传给下游的signal同时包含`spikes`和`inh_spikes`，下游可直接消费脉冲而非方向向量。后续神经接收器负责自己的输入延迟；测试已用一个额外神经元群体验证这种直接连接，但它不是正式MSTd模型。
单次调用异常后不保证回滚已推进的神经状态，应终止该次实验或显式reset。

## 输出与监视器

- `model_report.json`：backend、神经参数、tick数、初始化/计算时间、MSTd结果与边界说明。
- `monitors/v1/`：泊松采样前的连续响应，可逐帧重放。
- `monitors/mt_exc_rates/`、`mt_inh_rates/`：每帧实际脉冲数/时长，Hz。
- `monitors/mt_filtered_rates/`：因果低通率，Hz；没有空间池化。
- `monitors/mt_readout_activity/`：无量纲适配活动。
- `monitors/mt_*voltage/`、`mt_*recovery/`、`mt_*conductance/`：neural_state=true时保存的帧末状态。
- `monitors/spikes/`：spikes=true时记录所有细时间步，不受帧采样stride影响。三个.events文件各记录非零bin/neuron，字段为uint64-le tick、uint32-le neuron、uint32-le count，共16字节；不是CARLsim DAT。布局和时间见metadata.json。
- `estimates.jsonl`：MSTd逐帧估计，估计时间为帧区间终点。

监视器开关不改变网络计算和随机源。可调用 `model.set_spike_observer(callback)` 接收每步tick、步终点时间和事件副本，而不写文件。

## 后续研究

下一步是同一套组件上的脉冲MSTd、经任务验证的竞争/反馈，以及多速度/动态干扰/航向精度评估。当前实现不承诺复现CARLsim数值，也不等同于CD论文逐式复现。

## 查看MT响应与视频

补充同步 `examples/visualize_python_spiking.py`、更新后的 `examples/render_model_mt.py`，以及它依赖的已有 `examples/visualize_mt_gratings.py`。
静态图需要Matplotlib（项目plots可选依赖），MP4还需要PATH中的ffmpeg；不需要CARLsim或PyTorch。

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
# 如果缺少绘图库：python -m pip install 'matplotlib>=3.7'

# 自带右移光栅，自动保存原始脉冲和帧末神经状态：
python examples/validate_python_spiking.py --output experiments/scratch/manual__mt_demo_01__r01 --record-spikes

# 静态图。注意传d000子目录，而不是包含controls.json的父目录：
python examples/visualize_python_spiking.py experiments/scratch/manual__mt_demo_01__r01/d000 \
  --frame 30 --direction-deg 0 --rate-max 300 --output experiments/scratch/manual__mt_demo_plots_01__r01

# 静态图加输入/E/I/读出同步视频：
python examples/visualize_python_spiking.py experiments/scratch/manual__mt_demo_01__r01/d000 \
  --frame 30 --direction-deg 0 --rate-max 300 --video --output experiments/scratch/manual__mt_demo_video_01__r01
```

frame是从0开始的原输入索引，30表示第31帧；稀疏监视器要求该帧确实保存过。输出目录须不存在。
rate-max只控制显示色标，E/I均用0..rate-max固定范围；超过上限的显示会截顶，原始数值和CSV不改变。
direction-deg选择栅格图中的方向群体，默认0度右向；可用45/90/.../315。默认显示前1秒、固定均匀抽样的最多128个像素对应神经元。
换时间窗可加 `--raster-start-s 0.5 --raster-duration-s 0.5`。颜色图仍显示全部八方向。

主要输出：

| 文件 | 含义 |
|---|---|
| mt_exc_rates.png | 选中帧的八方向MT兴奋性实际脉冲率，Hz |
| mt_inh_rates.png | 选中帧的八方向MT抑制性实际脉冲率，Hz |
| mean_rates.png / mean_rates.csv | ROI内E/I等各方向随时间变化，CSV不裁剪数值 |
| spike_raster.png | 实际细时间步事件的栅格图；需要运行时spikes=true |
| voltage_samples.png | 中心像素所选方向E/I神经元的帧末电压采样；不是毫秒级放电波形；需要neural_state=true |
| mt_filtered_rates.png | 因果滤波率；不代表空间池化 |
| mt_readout_activity.png | MSTd输入适配活动；不是新增加的神经归一化回路 |
| mt_spiking.mp4 | 加--video时生成的同步视频，播放率等于输入帧率 |

自己的刺激：

```bash
python run_model.py --config configs/model_python_spiking.json \
  --stimulus /absolute/path/to/your_input.dat --max-frames 40 --output experiments/scratch/manual__mt_real_01__r01
python examples/visualize_python_spiking.py experiments/scratch/manual__mt_real_01__r01 \
  --frame 30 --video --stimulus /absolute/path/to/your_input.dat --output experiments/scratch/manual__mt_real_view_01__r01
```

视频逐帧与输入DAT做SHA256配对，不能用另一个相似刺激替代。记录跨机器移动后可用--stimulus指向原DAT的新位置。
视频要求every_n_frames=1。若只有连续V1重放记录且没有刺激SHA256配对依据，静态图可用，视频不绕过配对检查。
本机已实际导出32×32、40帧记录的PNG/栅格图及MP4，ffprobe确认1440×1230、40帧、20FPS。
