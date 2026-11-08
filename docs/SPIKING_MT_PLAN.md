# CARLsim MT 脉冲基线实施与验收（2026-11-05）

> 2026-11-06路线修订：本文保留为可选CARLsim离线对照后端的历史记录，不再作为自运动网络开发主线。Python统一时钟、MT神经元/突触及反馈接口已实现，见[Python主线运行说明](PYTHON_SPIKING_RUN.md)；脉冲MSTd仍待实现。

## 范围与模型定义

保留现有连续后端。新增 `mt_backend=carlsim` 的离线两阶段执行：
图像 → Python V1（或完整连续 V1 记录）→ CARLsim 泊松输入 → Izhikevich MT E/I → 窗口率 → 现有连续 MSTd。
直接使用真实 CARLsim 库，不另写近似积分器；缺少可执行文件时明确失败，不静默退回连续 MT。
本版不是完整脉冲 MSTd，也不是 CD 论文复现。MT 不新增空间池化、归一化群体或反馈。

### 固定的基线

- 原始 SFYX 输入为 3×28×H×W，按帧在 frame_ms 内保持连续率。
- 使用 configs/v1_projection.json 的具名 28×8 矩阵；该配置取自 v1toCDS1_LoadSimulation.cpp，不保证等于 main 文件或历史加载快照。
- 三尺度都参与，要求原配置 scale_weights=[1,1,1]、gain=1。
- E/I 神经元都用 (a,b,c,d)=(0.02,0.2,-65,8)；setConductances(true)，积分与电导常量继承所链接库的默认配置，随库版本登记。
- V1→E: 1.5*max(W,0)；V1→I: 1.5*max(-W,0)；I→同方向 E 权重1、延迟1ms；固定突触。
- 使用旧版三群体布局：V1=H×W×84，MT E/I各H×W×8，避免默认库96群体/128连接组上限。最初拆成100群体的优化方案经真实库测试发现超限，已取消。
- 默认 `carlsim.callback_layout=legacy_mapped`，匹配本地修改过的内核（回调传入filter/direction，connected已表示同位置）；上游标准库必须显式设置 `standard`（回调传入群体内神经元编号）。实际连接数在setup后逐连接组核查，契约不匹配拒绝完成。
- 首版保留全配对扫描的构网成本，不能宣称构网优化已完成。两个V1→MT投影各检查 (84P)*(8P) 个候选，80²下成本很高。
- 神经元状态跨帧连续。保存三个群体的原生SpikeMonitor DAT（v1.dat/exc.dat/inh.dat）；不因静默尾段缩短时长。原生脉冲文件始终保存；新后端保存E率、I率、适配活动，按every_n_frames采样；原连续后端的分阶段监视开关不用于这些新文件。
- `carlsim.inhibition_enabled=false` 仅用于诊断性消融；默认true。不同拓扑即使种子相同也可能改变输入随机序列，消融使用多种子统计才可作正式比较。
- 输入率必须在缩放后[0,1000]Hz内；超限拒绝，不静默截断。v1_rate_scale默认1，改变时需另标模型配置。
- 原生E/I率是观测脉冲率。MSTd适配活动 r_d/(1Hz+Σr) 仅是读出适配器，不是神经网络归一化；保留完整八方向率与I率。

## 实施步骤

1. 新增独立 C++ bridge 与 CMake 构建入口，对接已安装CARLsim。
2. Python验证输入、导出V1、启动bridge、严格读取脉冲并接入MSTd。记录配置/输入/权重/可执行文件哈希、构网/仿真/读出时间与实际连接数。
3. 配置与统一CLI接入，保留旧默认；新增测试，不覆盖已有用户修改和旧实验。
4. 本机验证协议、读出、集成及真实CARLsim CPU小网；GPU和历史运行库/快照对照在原服务器执行。

## 验收指标与判定

| 功能 | 指标 | 首版判定 |
|---|---|---|
| 索引/接线 | 非零W对应边数、方向/位置、E/I符号、延迟 | 预期连接数 P*(84*8中按符号非零数+8)，位置零串扰；小网逐连接检查 |
| 泊松输入 | 固定率下计数均值/方差、通道独立性、0率事件数 | 0率无事件；非零率多种子均值相对目标的标准化偏差≤5；Fano接近1须结合CARLsim不应期与离散采样解释 |
| 神经元与突触 | 固定脉冲输入下膜电位、E/I事件、延迟 | 同一库/分组/驱动的独立小网对照；确定性条件逐事件相同、膜电位容差预先固定 |
| 抑制通路 | E/I输出率与抑制关闭对照 | 专用正负通道刺激下I有效激活，抑制降低E响应；不得要求所有自然刺激单调降低 |
| 文件与时间 | 事件顺序/编号/范围、窗口计数、静默尾段 | 严格校验全部事件；50ms单事件=20Hz；边界事件归下一窗 |
| 复现 | 相同库/模式/配置/种子重复结果 | 同环境检查事件哈希；跨CPU/GPU用多种子统计，不预设逐事件相等 |
| 局部方向 | 八方向光栅/随机纹理的调谐、环形误差、静态误报、停止/切换延迟 | 在开发集冻结阈值后独立测试；报告失败，不以峰值与群体向量互相替代 |
| MSTd闭环 | 航向MAE/P95、有效覆盖率、静止误报、跟踪延迟 | 同刺激对照连续版本与旧MT；本机夹具仅验证接口，不宣称自运动精度 |
| 规模性能 | 32²/80²构网秒数、仿真秒数、内存/显存峰值、事件存储 | 报告实测，不预设实时倍数；80²先评估候选配对量，不盲目长跑 |

协议/接口通过≠原CARLsim完整复现通过。本机CPU测试与历史GPU库/快照一致性验收分别记录。

## 构建与运行

在原Linux CARLsim环境使用已安装的头文件与库（本仓库不改原工程）：

```bash
cmake -S native/carlsim_mt -B build/carlsim_mt \
  -DCARLSIM_INCLUDE_DIR=/path/to/carlsim/include \
  -DCARLSIM_LIBRARY=/path/to/libcarlsim.so
cmake --build build/carlsim_mt
# 静态库额外依赖可通过 -DCARLSIM_EXTRA_LIBRARIES='cudart;curand;pthread' 指定，按本机安装调整。
# Windows导入库另用 -DCARLSIM_RUNTIME_LIBRARY=/path/to/carlsim.dll 保存实际运行库哈希。
# 修改版内核使用legacy_mapped；标准上游内核改配置为carlsim.callback_layout=standard。
# 设置 configs/model_spiking.json 的输入和 executable 为真实绝对路径后：
python -m mstd simulate --config configs/model_spiking.json
# 真实引擎受控小网验收；加--grating可跑32×32右移光栅完整链路：
python examples/validate_spiking_mt.py --executable /absolute/path/to/carlsim_mt \
  --output experiments/validation/manual__spiking_check_01__r01 --grating
```

先32×32短刺激，逐步扩大；本版两阶段离线文件中转不适合直接宣称在线实时处理。

Windows本机CPU构建原库遇到缺失ExecutionStopwatch声明，通过编译器强制包含原有execution_stopwatch.h解决，未编辑旧源码。复查构建参数见build/legacy_20261107/carlsim_cpu_build/CMakeCache.txt。桥接入口设置二进制默认文件模式，避免旧库的"w+"在Windows插入CRLF破坏DAT。
