# MT方向竞争第二阶段

日期为2026-11-16。已完成独立竞争抑制群、正常模型接口、事件与状态监视器、四组消融入口及本机检查。GPU专项与完整刺激验收待服务器执行。

## 1 开发前Git检查点

开发前代码已提交并推送到origin/main，提交为b8f961caafb93cfe6908d0bfa5443553c1ec7b5e。提交包含当时的源码、配置、测试、文档和轻量索引。推送经现有SSH认证完成，并核对远程main与本地提交相同。

本阶段实现作为该检查点之后的工作区改动交付。

## 2 神经回路

每个位置增加8个方向的竞争抑制神经元J。MT-E方向d的脉冲，经独立延迟和兴奋突触驱动J_d。J群输出经方向权重矩阵和另一段延迟，形成MT-E其他方向的抑制电导。

每个神经tick开始先读取全部到达信号，再更新E、原有I与新增J群。三个群完成更新后，统一将新脉冲写入各自延迟线。两段延迟分别可配置，默认各1tick。神经元发放过程还会产生自身的响应时间。

原有I群仍负责原V1负权重通路。新增J群独立记录，便于检查方向竞争的贡献。

角度差采用圆周距离。angular模式的权重与1−cos(方向差)成正比，矩阵对角为0，各接收行归一化。opponent模式只保留180°对向连接。两个模式分别可测量。

J使用现有IzhikevichPopulation类，参数为a=0.1、b=0.2、c=−65、d=2。模型报告保存这些参数。神经电导为有效模型单位。

2009年ViSTARS提供方向竞争与方向相关连接的机制依据。新增脉冲J群、两段离散延迟和初始突触权重属于本项目工程适配，详细借鉴关系见MT_MSTD_CIRCUIT_DEVELOPMENT_V01.md。[原论文](https://sites.bu.edu/steveg/files/2016/06/BroGroMin2009.pdf)

## 3 配置

新增配置字段为mt_competition。

```json
{
  "enabled": true,
  "mode": "angular",
  "input_weight": 0.25,
  "inhibition_weight": 0.2,
  "tau_exc_ms": 5.0,
  "tau_inh_ms": 10.0,
  "input_delay_ticks": 1,
  "output_delay_ticks": 1
}
```

input_weight控制E→J驱动，inhibition_weight控制J→E抑制。二者均为工程初始候选。验收配置使用1ms tick和32积分子步，空间参数沿用半径2像素、增益0.05。

本机先检查了E→J权重1.0。该候选在单纹理条件中出现约1100Hz的最大J帧率，J平均率对16/32子步的相对变化约2.13%。随后选择较小驱动0.25，并补充32/64检查。

在同一32×32纹理输入下，较小驱动候选的32/64平均率变化为MT约0.15%、J约0.68%。这一结果支持当前小输入范围的数值参考，完整刺激及GPU范围继续单独验收。两个驱动候选及积分检查均保留原生结果和源码快照。

## 4 本机检查

- 65项相关检查通过，包含14项新增竞争与CLI计划检查。
- 检查覆盖矩阵归一化、方向置换与镜像、两段延迟、突触衰减、位置局部性、更新顺序、零输入、重置和跨帧状态保持。
- 受控脉冲输入下，竞争能够抑制较弱对向分量，对称的双方向输入保持对称。
- 关闭竞争时，空间关闭及空间开启两个条件均逐位复现开发前检查点的100-tick状态摘要。
- 零输出增益的E/I活动与基线一致。
- J的原始事件能够重建所保存的J放电率，监视器开关及V1重放检查通过。
- NumPy与PyTorch CPU的固定事件对照通过，run_model.py的PyTorch CPU图像入口通过。
- CUDA专项待服务器执行。本机环境为PyTorch 2.15.0+cpu。

## 5 小型四组对照

候选运行为experiments/validation/20261116_222616__mt_competition_controls__py_spike_cpu__sub32__s0043__r01。每条件32×32、24帧，前20帧预热，评价4帧。三个候选重放baseline的同一V1录制，脉冲种子均为43。

| 模型 | 静态MT平均率 | 向右运动整体方向MAE | 向右运动活跃像素MAE |
|---|---:|---:|---:|
| baseline | 5.318Hz | 1.821° | 44.018° |
| spatial_only | 5.398Hz | 1.681° | 43.788° |
| competition_only | 5.077Hz | 1.862° | 44.763° |
| spatial_competition | 5.153Hz | 1.523° | 44.477° |

仅竞争组静态平均率下降约4.53%，联合组下降约3.11%。静态MSTd几何接受均为0/4帧。

联合组整体方向误差在此样本中降低，局部方向误差仍高于基线。四组共同活跃像素的MAE分别为44.019°、43.713°、44.631°、44.326°，共同覆盖率约99.81%。当前结果尚未确认局部精度收益。

完整审计见outputs/comparisons/mt_competition/20261116__mt_competition_stage2__v01/review.json。该审计核对120组监视器、输入哈希、分组开关及原有I群的一致性。

## 6 源码同步清单

将以下工作区改动同步到服务器code目录中的对应位置。

```text
src/mstd/mt_circuit.py
src/mstd/spiking_mt.py
src/mstd/model.py
src/mstd/monitors.py
tools/run_mt_spatial_controls.py
tools/run_mt_competition_controls.py
tools/render_mt_spatial_controls.py
configs/model_python_spiking_competition32_v01.json
configs/model_python_spiking_spatial_competition32_v01.json
tests/test_mt_competition.py
tests/test_mt_competition_controls.py
docs/MT_DIRECTION_COMPETITION_STAGE2_V01.md
```

入口还复用已有src/mstd及examples中的录制、读出和绘图代码，直接同步当前源码目录可保持依赖一致。

## 7 服务器运行

先执行新增专项检查。在检测到CUDA设备时，固定事件检查会覆盖CUDA路径。

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_mt_competition*.py' -v
```

查看GPU小型四组计划并执行。

```bash
python tools/run_mt_competition_controls.py \
  --backend torch --device cuda --quick-check --plan-only

python tools/run_mt_competition_controls.py \
  --backend torch --device cuda --quick-check
```

专项和小型运行检查后执行完整矩阵，共33个条件、每条件4次模型运行。脚本自动生成协议夹具并登记索引。

```bash
python tools/run_mt_competition_controls.py \
  --backend torch --device cuda --cases static,texture \
  --substeps 32 --seed 43 --gain 0.05 --radius 2 \
  --competition-input-weight 0.25 --competition-weight 0.2 \
  --competition-mode angular
```

需要对向连接对照时使用competition-mode opponent。原KITTI冻结图像及动态录制可通过cases kitti继续配对运行，该入口读取父实验记录的帧间隔与内参。

## 8 输出及MP4

脚本打印全新的experiments/validation/<运行ID>目录，运行ID包含mt_competition_controls。每个条件保存baseline、spatial_only、competition_only、spatial_competition四个子目录及comparison.json，根目录保存plan.json、summary.json、artifact.json。

新增监视器包括mt_competition_rates、mt_competition_voltage、mt_competition_recovery、mt_competition_exc_conductance和mt_competition_inh_conductance。后者表示J→MT-E抑制。mt_inh_conductance记录作用于E的总抑制电导。

启用monitors.spikes时，还保存competition.events及对应metadata。其计数可用于毫秒级时序检查。

静态及纹理夹具的四组MP4可按以下命令生成，将RUN替换为本次运行目录。

```bash
RUN=experiments/validation/<本次运行ID>
python tools/render_mt_spatial_controls.py \
  --run "$RUN" --case texture_d000_s1 --four-way
```

MP4保存在outputs/comparisons/mt_competition/<运行ID>__<条件>__rates__v01/mt_spatial_comparison.mp4。四排从上至下依次为基线、仅空间、仅竞争、联合，使用共同Hz色标和相同八方向顺序。

本阶段代码及基础验证已经完成。功能评价继续同时检查静态响应、各方向/速度的整体与局部误差、覆盖率、多个运动方向保持及数值敏感性，随后再据结果选择参数。
