# Python脉冲MT响应分析流程

日期：2026-11-08。适用后端：`mt_backend=python_spiking`。
本文是操作手册，不代表任何一段自然视频已经完成分析或通过指标验收。
项目根目录是code/；目录与命名遵循FILE_ORGANIZATION.md。

## 1 分析目标与推荐顺序

按以下顺序进行，不要只凭热图颜色判断模型好坏：

1. 确认运行、输入、参数、保存帧数和数据完整性。
2. 读取MT兴奋/抑制发放率，观察方向和空间响应。
3. 分别分析原始发放率与因果滤波活动的时间变化。
4. 使用实际脉冲事件检查时间结构，并核对帧率统计。
5. 联合膜电位、恢复变量、电导定位机制问题。
6. 检查MT送往当前MSTd的活动表征。
7. 对照刺激或独立运行，区分功能描述、准确性与因果结论。
8. 保存分析参数、数值结果、结论和未解决问题，登记关联。

日常最小分析使用`mt_exc_rates`、`mt_inh_rates`和对应metadata；研究时序再读events，机制排错再读状态。

## 2 先定位一次运行

正式响应目录：

```text
outputs/recordings/<dataset>/<run_id>/
  artifact.json
  model_config.json
  model_report.json
  frames.csv
  estimates.jsonl
  monitors/...
```

scratch和validation运行应按registry找到实际位置，不为分析移动或覆盖原始数据。
先查`experiments/registry.csv`定位run_id，再查`input/registry.csv`和刺激包元数据定位输入。

记录以下信息：

| 项目 | 文件/字段 |
|---|---|
| 输入ID、哈希、父运行 | artifact.json；与model_report.input核对 |
| 后端和设备 | model_config的mt_backend/backend/device；model_report.model |
| 神经步长、积分子步、种子、突触参数 | model_config.spiking |
| 图像时间间隔、帧数限制 | model_config.frame_ms/max_frames |
| 实际处理帧数与完成状态 | model_report.frames/complete |
| 保存间隔和启用的监视器 | model_config.monitors、model_report.monitors、各metadata |
| 原始输入尺寸/帧数 | DAT头和字节数；可用mstd.calibration.inspect_stimulus |
| 输入帧时序 | 配套.dat.json/manifest及本次frame_ms，不从DAT头猜帧率 |

正式分析要求`complete=true`，并确认运行确实使用python_spiking。失败/部分实验可做排错，但标为partial/failed，不能伪装成完整结果。
SHA256、尺寸或帧序不匹配时先定位配对问题，不用翻转、平移时间或更改增益强行匹配。

## 3 文件含义速查

以下目录均相对`monitors/`。数组目录包含`metadata.json`和`values.f32`。

| 目录 | 含义 | 单位/采样 |
|---|---|---|
| mt_exc_rates | MT-E每帧实际脉冲计数/帧时长，主要方向响应 | Hz，帧区间统计 |
| mt_inh_rates | MT-I每帧实际脉冲计数/帧时长 | Hz，帧区间统计 |
| mt_filtered_rates | MT-E脉冲bin率的因果指数低通迹 | Hz，帧末采样 |
| mt_readout_activity | 滤波率的适配活动r_d/(1Hz+Σr) | 无量纲，帧末采样 |
| mt_exc_voltage / mt_inh_voltage | E/I群体的膜电位v | mV，帧末采样 |
| mt_exc_recovery / mt_inh_recovery | E/I群体的恢复变量u | 模型单位，帧末采样 |
| mt_exc_conductance | 驱动MT-E的总兴奋电导，含启用时的反馈分量 | 有效模型单位，帧末采样 |
| mt_inh_conductance | 驱动MT-E的总抑制电导，含启用时的反馈分量 | 有效模型单位，帧末采样 |
| spikes/exc.events | MT-E细时间步事件 | 稀疏tick/neuron/count记录 |
| spikes/inh.events | MT-I细时间步事件 | 同上 |
| spikes/v1.events | V1泊松输入事件 | 同上，群体布局不同 |

两个conductance文件不是“E/I两个群体各自的全部电导”。它们都描述施加到MT-E的作用；当前未单独保存MT-I的输入兴奋电导。
`mt_filtered_rates`不是空间池化；`mt_readout_activity`不是新增的神经归一化回路。
电压和电导只有帧末样本，不能据此重建毫秒级放电波形或精确突触峰值。
spikes=true才有events，neural_state=true才有状态监视器。缺失不等于神经元静默。

## 4 第一步：正确加载数组

安装/激活已有NumPy环境，从code/运行自己的分析脚本。下面是只读示例，把run路径替换为实际目录。

```python
import json
import math
from pathlib import Path
import numpy as np

run = Path('outputs/recordings/daria_walk/替换为实际运行ID')
report = json.loads((run / 'model_report.json').read_text(encoding='utf-8'))
config = json.loads((run / 'model_config.json').read_text(encoding='utf-8'))
assert report['complete'] is True
assert report['model']['mt_backend'] == 'python_spiking'

def load_array(name):
    folder = run / 'monitors' / name
    meta = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
    assert meta['schema'] == 'mstd-layer-monitor-v1'
    assert meta['complete'] is True
    assert meta['dtype'] == 'float32-le' and meta['layout'] == 'TDYX'
    path = folder / meta['file']
    assert path.stat().st_size == math.prod(meta['shape']) * 4
    values = np.memmap(path, dtype='<f4', mode='r', shape=tuple(meta['shape']))
    assert values.shape[1] == 8
    assert len(meta['sample_indices']) == len(values)
    assert len(meta['timestamps_s']) == len(values)
    for frame in values:  # 不一次复制整个大数组
        assert np.isfinite(frame).all()
    return meta, values

exc_meta, exc = load_array('mt_exc_rates')
inh_meta, inh = load_array('mt_inh_rates')
assert exc_meta['units'] == inh_meta['units'] == 'Hz'
assert exc.shape == inh.shape
assert exc_meta['sample_indices'] == inh_meta['sample_indices']
assert np.allclose(exc_meta['timestamps_s'], inh_meta['timestamps_s'])
assert all((frame >= 0).all() for frame in exc)
assert all((frame >= 0).all() for frame in inh)

indices = np.asarray(exc_meta['sample_indices'])
times = np.asarray(exc_meta['timestamps_s'])
angles = np.degrees(exc_meta['directions_rad']) % 360
assert np.allclose(times, (indices + 1) * config['frame_ms'] / 1000)
```

布局为`[保存样本T, 方向D, y, x]`，不是`[T,D,x,y]`。单位必须读metadata，不按文件名猜测。
当前常用方向顺序是0、315、270、225、180、135、90、45度，但代码应读取directions_rad。
部分状态metadata没有方向标签，联合分析时先核对形状/帧号/时间，再使用同运行E率监视器的方向约定。
在Windows长驻交互环境中，用完memmap可调用`exc._mmap.close()`等释放句柄；不要在仍使用数组视图时关闭。

## 5 第二步：空间响应与方向分布

先选一个原输入帧，再通过sample_indices定位，不能把保存样本位置直接当原帧号。

```python
input_index = 30  # 零起点，第31帧
matches = np.flatnonzero(indices == input_index)
if len(matches) != 1:
    raise ValueError('该输入帧没有保存，检查监视器stride')
sample = int(matches[0])
maps = exc[sample]              # [8,H,W]
pixel_rates = maps[:, 20, 10]   # 像素x=10,y=20；先确认坐标未越界
```

同时看输入画面与E/I八方向图，检查：响应位置、边缘、运动边界、是否始终固定某一通道、是否全部静默或普遍高放电。
边界补偿和滤波器历史会影响空间图；强响应不是物体分割，亮度边缘也不等于运动边界。

固定ROI后计算方向曲线。以下排除4像素边缘，仅为可复现分析选择，不是生理标准：

```python
b = 4
_, _, H, W = exc.shape
assert 2 * b < min(H, W)
def roi_means(values):
    return np.stack([
        frame[:, b:H-b, b:W-b].mean(axis=(1, 2), dtype=np.float64)
        for frame in values
    ])

exc_mean = roi_means(exc)  # [T,8]
inh_mean = roi_means(inh)
total = exc_mean.sum(axis=1)
fractions = np.divide(exc_mean, total[:, None],
                      out=np.full_like(exc_mean, np.nan), where=total[:, None] > 0)
winner_index = exc_mean.argmax(axis=1)
winner_deg = np.where(total > 0, angles[winner_index], np.nan)

z = (exc_mean * np.exp(1j * np.radians(angles))[None, :]).sum(axis=1)
concentration = np.divide(np.abs(z), total,
                          out=np.full_like(total, np.nan), where=total > 0)
vector_deg = np.where((total > 0) & (np.abs(z) > 1e-12),
                      np.degrees(np.angle(z)) % 360, np.nan)
```

解释：

- winner_deg：最强通道，不等于分布式群体向量方向。
- concentration：方向分布的集中程度，不是“估计正确概率”；对向响应会抵消。
- 全图/ROI平均可能把局部不同运动混在一起，必要时对背景/局部区域分别分析。
- 静默时方向无定义；低集中度时群体向量不稳定，应报告低证据，不强制赋0度。
- 全局平均后取方向与先求每像素方向再平均不是同一种运算，报告中写清方法。

## 6 第三步：时间变化与输入对齐

画exc_mean/inh_mean的八方向时间曲线；横轴使用MT metadata.timestamps_s（帧末证据可用时间）。
原输入帧i对应区间`[i*frame_ms,(i+1)*frame_ms]`的驱动，响应在区间终点可用；frames.csv的timestamp_s为帧起点，available_timestamp_s为终点，不能混作同一时间。
记录启动、预热、稳定区间和停止/方向切换时刻。独立序列才reset，不能为让曲线好看而逐帧清状态。

原始E率、滤波率和适配活动分别画图：

- 原始E率保留帧内计数波动。
- 滤波率有trace_tau_ms控制的时间平滑，是帧末迹，不等于该帧脉冲数/时长。
- 适配活动压缩幅度且受同位置全部方向影响，没有Hz单位。

滤波延迟和V1历史延迟保留在分析中；不自动平移曲线掩盖迟滞。
需要检测启动/停止延迟时，事先在开发集确定基线、响应阈值和持续时间判据，再冻结后测试；没有刺激切换标注就不输出“准确延迟”。

## 7 第四步：解析实际脉冲事件

每条记录16字节：tick:uint64-le、neuron:uint32-le、count:uint32-le。不是CARLsim DAT协议。

```python
folder = run / 'monitors' / 'spikes'
spike_meta = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
assert spike_meta['schema'] == 'mstd-python-spike-count-events-v1'
assert spike_meta['complete'] is True and spike_meta['record_bytes'] == 16
dtype = np.dtype([('tick', '<u8'), ('neuron', '<u4'), ('count', '<u4')])
group = 'exc'  # 换成inh可分析MT-I
path = folder / spike_meta['files'][group]
n = spike_meta['records'][group]
assert path.stat().st_size == n * dtype.itemsize
events = (np.memmap(path, dtype=dtype, mode='r', shape=(n,))
          if n else np.empty(0, dtype=dtype))
D, H, W = spike_meta['shapes'][group]
dt_ms = spike_meta['dt_ms']

# 以下以块检查，避免复制整个大事件文件
previous_tick = -1
total_spikes = 0
for begin in range(0, n, 65536):
    block = events[begin:begin+65536]
    ticks = block['tick'].astype(np.int64)
    assert ticks[0] >= previous_tick and np.all(np.diff(ticks) >= 0)
    assert np.all(block['tick'] < spike_meta['ticks'])
    assert np.all(block['neuron'] < D * H * W)
    assert np.all(block['count'] > 0)
    previous_tick = int(ticks[-1])
    total_spikes += int(block['count'].sum(dtype=np.uint64))

# 示例：只解码前1000行；大文件按块或时间窗筛选
sample_events = events[:1000]
time_s = (sample_events['tick'].astype(float) + 1) * dt_ms / 1000
neuron = sample_events['neuron'].astype(np.int64)
x = neuron % W
y = (neuron // W) % H
direction_index = neuron // (W * H)
```

`len(events)`是非零bin/神经元记录行数，不是总脉冲数；总数必须累加count。
V1事件布局为`[3,28,H,W]`，不能套用MT的方向解码；V1泊松bin允许多个事件，MT在积分子步内也可能累积多次放电。

### 用事件核对帧内率

当前Python记录以tick所属的图像帧累计计数。若50ms/帧、1ms/tick，tick0..49都属于原输入帧0；tick49的标记时间恰好为50ms。
因此用`tick // ticks_per_frame`分帧，不能直接用浮点事件终点时间向下取整。不要照搬旧CARLsim DAT的时间窗边界约定。

```python
ticks_per_frame_float = config['frame_ms'] / dt_ms
ticks_per_frame = round(ticks_per_frame_float)
assert np.isclose(ticks_per_frame_float, ticks_per_frame)
target_input_index = int(indices[0])
counts = np.zeros(D * H * W, dtype=np.uint64)
for begin in range(0, n, 65536):
    block = events[begin:begin+65536]
    mask = block['tick'] // ticks_per_frame == target_input_index
    chosen = block[mask]
    np.add.at(counts, chosen['neuron'], chosen['count'].astype(np.uint64))
reconstructed = counts.reshape(D, H, W) / (config['frame_ms'] / 1000)
np.testing.assert_allclose(reconstructed, exc[0], rtol=1e-6, atol=1e-5)
```

此例group=exc对应exc数组，group=inh时改为inh数组。只核对选中的一个保存帧，不为大视频分配完整T×D×H×W计数副本。
先通过这一检查再研究栅格、ISI、Fano等。ISI精度受dt与bin内聚合限制；统计Fano/变异系数要说明窗口、平稳性和重复试验，不将非平稳自然视频的计数方差直接解释为泊松性。

## 8 第五步：E/I与神经元机制排查

按相同方向、位置、帧号和时间联合看：E/I率、脉冲、电压v、恢复变量u，以及施加到E的兴奋/抑制电导。

| 现象 | 下一步检查 |
|---|---|
| 所有方向静默 | V1输入是否有事件、正权重驱动、电导/阈值、保存开关 |
| E异常高放电 | 输入增益、input_weight、时间步/积分子步、电导及恢复变量 |
| I高而E弱 | 是否符合该方向负权重通路；结合输入和延迟，不只看全图均值 |
| 转向后旧方向残留 | V1历史、突触衰减、trace_tau_ms、实际神经状态与反馈 |
| 空间图整体均匀 | 原输入/投影是否已经均匀、索引/ROI是否正确，不先归因于显示 |
| 电压图没有尖峰 | 帧末采样会漏掉阈值与复位过程，回看真实事件；需波形时另设计细步电压记录 |

重要区别：I群体的放电率不是抑制电流，g_inh也不是抑制电流。作用还依赖E膜电位与反转电位；帧末g/v不足以精确还原整帧突触电流。
E/I率比可作描述，但不是回路平衡的充分证据，更不能证明因果抑制。

如需证明抑制的作用，建立独立消融运行：保持同一输入/种子/参数，仅改变指定抑制机制，并验证V1完整事件一致，再比较E率和任务结果。
改变模型参数必须创建新run_id，原结果不覆盖。当前框架的直接反馈控制器和神经参数需按各自接口设置，不把旧CARLsim配置键当成Python配置键。

## 9 第六步：检查供MSTd使用的表征

用上述load_array分别读取mt_filtered_rates和mt_readout_activity，确认：

1. 同一次运行、相同sample_indices/shape/timestamps。
2. 滤波率有限且非负；适配活动有限、非负，各像素八方向之和小于或数值上接近1。
3. 用滤波率计算`r / (1 + r.sum(axis=方向))`，与保存的适配活动按浮点容差比较。
4. 静默/低证据时不强行解释方向；比较原始E率和滤波表征的时序与方向差异。

当前MSTd每个神经时间步接收因果活动更新；帧末监视器不能用于推断整帧内部的所有MSTd输入。
MT中间层方向表征不是已校准光流速度；数组中的方向向量不能直接解释为px/s或m/s。
现有MSTd仍是连续适配器，不用MT图像证明脉冲MSTd或完整自运动模型已验证。

## 10 生成图和视频的命令

在项目根目录，替换dataset、run_id、输入路径；不要把占位符直接当实验目录名。

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python examples/visualize_python_spiking.py \
  outputs/recordings/<dataset>/<run_id> \
  --frame 30 --direction-deg 0 --rate-max 300 \
  --video --stimulus <对应输入DAT当前路径> \
  --output outputs/visualizations/<dataset>/<run_id>/mt_spikes__v02
```

若v02已存在则使用未占用版本；只看图片去掉--video。frame为零起点原输入编号，必须确实保存过。
查看时间窗可加`--raster-start-s 0.5 --raster-duration-s 0.5`。栅格默认是指定方向中固定抽样的最多128个像素对应神经元，不是完整群体。
PNG/MP4需要Matplotlib；MP4另需ffmpeg。输入DAT做SHA256配对，不能换成相似视频。

优先查看：mt_exc_rates.png、mt_inh_rates.png、mean_rates.png、spike_raster.png，最后再看状态图。
mean_rates.csv包含ROI统计，rate-max只裁剪显示色标，不改CSV和原始数值；颜色亮不等于跨单位/跨色标响应可比。
可视化工具不会自动为手工指定的新展示目录登记artifact/registry；通过自然刺激封装首次生成时会登记，手工重绘须按下一节补登记。

## 11 有无真值时分别能得出什么结论

| 条件 | 可报告 | 不应直接报告 |
|---|---|---|
| 已知方向光栅 | 调谐曲线、峰值/群体向量方向误差、稳定性 | 一般二维光流准确率；光栅有孔径歧义 |
| 已知平移随机纹理 | 同条件方向误差、适用速度范围、跨种子分布 | 未校准的速度EPE |
| 有标注切换/停止刺激 | 按预先冻结判据计算的启动/停止/切换延迟 | 任意移动时间曲线后的“零延迟” |
| Daria/paragliding/snowboard自然刺激无运动真值 | 空间/方向分布、时序、静默和高放电比例、不同ROI描述 | 方向准确率、相机航向精度或动态物体分割准确率 |
| 开启/关闭某机制的受控对照 | 指定条件的响应/任务变化及代价 | 单凭I高E低断言生物机制正确 |

有方向真值时用环形误差`abs((pred-true+180)%360-180)`；同时报告有效覆盖率和静默/低证据比例，不靠拒绝困难样本制造低误差。
没有真值时，最强通道只表示“模型偏好”，不是“正确答案”。

## 12 建议保存的分析结果和命名

单运行分析：

```text
outputs/visualizations/<dataset>/<run_id>/mt_analysis__v01/
  artifact.json
  analysis_config.json
  analysis_report.md
  frame_summary.csv
  pixel_summary.csv       # 如分析了特定像素/ROI
  spike_statistics.csv   # 如进行了事件统计
  figures/...
```

这是建议的分析交付格式，不表示现有绘图脚本会自动生成这些分析文件。
analysis_config记录ROI/像素、帧区间、预热、方向约定、时间窗、筛查阈值、色标、父响应路径/哈希和分析脚本版本。
frame_summary建议字段：input_index、frame_end_s、八方向E/I均值、总活动、winner_deg、vector_deg、concentration、静默标记。需要阈值的指标额外记录阈值来源。
analysis_report写明输入/run_id、方法、发现、证据、异常、尚未解决的问题和下一步，不把completed自动写成passed。

新增analysis/view目录须写mstd-artifact-v1的artifact.json，关联原run_id、stimulus_id、输入哈希；在experiments/registry.csv新增一行，单运行展示/分析可登记为visualization并在notes注明用途。
跨运行结果放outputs/comparisons/<dataset>/<comparison_id>/，列出所有父运行，先确认相同输入、像素几何、时间和参数；不同网格不直接逐像素比较。
原生响应保留在recordings，不复制到展示目录作为唯一数据源，不覆盖或移动原实验。

## 13 每次分析结束的检查清单

- [ ] 找到了正确运行、输入ID及输入哈希，已确认后端与实际帧数。
- [ ] 核对数组大小、dtype、布局、单位、方向、保存帧号和时间戳。
- [ ] 空间图与原输入同步，ROI和预热区间有记录。
- [ ] 静默方向保留为无定义，未把集中度当正确概率。
- [ ] 需要时已用events重建所选帧的实际率，保留count和tick边界约定。
- [ ] 状态解释没有把帧末样本当完整波形，也没有把电导当电流。
- [ ] 区分了原始率、滤波率和无量纲读出；未标成物理速度。
- [ ] 准确性/延迟/因果结论有相应真值或受控对照；自然视频无真值时只作描述。
- [ ] 保存分析参数、父运行与源码依据，并登记artifact和registry。

相关实现：src/mstd/monitors.py、spiking_mt.py、model.py、neural.py；运行说明：PYTHON_SPIKING_RUN.md、NATURAL_SPIKING_RUNS.md。
