# 第三阶段MSTd时间竞争

2026-11-17。已完成动力学配置、消融入口、理想输入协议、保存MT的帧采样重放、固定MT的端到端入口及曲线MP4。MSTd采用现有连续率状态方程，参数选择继续通过独立输入验收。

## 根据第二阶段结果安排第三阶段

第二阶段完整矩阵中，仅空间组在32个运动条件的局部误差均有小幅改善，仅竞争组在31个条件高于基线。本阶段端到端入口选用原MT基线与仅空间分支，竞争分支继续保留在第二阶段消融工具中。

## 配置与消融

新增mstd_dynamics配置，字段为tau_s、max_step_s、gain、recurrent_gain、inhibition、min_evidence、min_margin，单位分别按秒及现有无量纲模型定义。默认值与开发前相同，100tick的默认MSTd活动及MT脉冲摘要逐位复现。

| 名称 | 时间常数 | 自激励 | 群间抑制 |
|---|---:|---:|---:|
| reference | 80ms | 0.5 | 1 |
| input_only | 80ms | 0 | 0 |
| no_self | 80ms | 0 | 1 |
| no_lateral | 80ms | 0.5 | 0 |
| tau40 | 40ms | 0.5 | 1 |
| tau160 | 160ms | 0.5 | 1 |

input_only保留输入饱和项，属于无递归的输入动力学。所有设置使用相同模板、输入、有效性判据及角度读出。模板覆盖水平−20°至20°，间隔5°，采用local_quadratic读出。

## 理想MT协议

默认32×32，每条件100帧、50ms/帧，内部每1ms更新，随机种子43。六类输入为网格外航向保持、−7.5°到10°阶跃、−12°到12°缓慢扫动、半场短时遮挡、均匀平移拒绝控制、随机方向干扰后停止。

输入是直接送入MSTd的无量纲方向证据向量，保存为input_vectors.f32及input.json。元数据包含TCYX布局、时间间隔、合成相机内参、逐帧真值和哈希。该文件在input索引中标为fixture，channels=2表示向量分量。

正式alpha_x_deg受有效性门控。candidate_heading_deg用于离线无门控误差分析，输出有效率与候选误差一起报告。invalid_target_frames字段统计无航向目标窗口中被接受的帧数。切换和恢复时间采用误差≤2.5°且连续3帧有效的首次进入时刻，按帧末时间计算，属于离线回看指标。容差2.5°及3帧窗口是本次工程评价约定。

## 本机结果

71项相关检查通过，包含6项新增配置、方程一致性、状态、协议及指标检查。CUDA专项待服务器执行。

理想保持条件reference的无门控平均误差约0.044°，目标窗口有效率100%。阶跃条件reference稳定进入时间50ms，tau160为100ms。噪声后停止条件的reference候选平均误差约0.132°，tau160约0.113°。停止后活动衰减，均匀平移和停止窗口的接受计数为0。

这些结果覆盖本轮合成协议，尚需独立随机种子、更强干扰、GPU和真实图像输入评价。时间常数继续作为对照候选。

正常图像入口完成12次两帧试跑，原MT基线与仅空间分支分别跨六种动力学设置。每个MT分支的读出哈希在六次运行间一致。该试跑验证配置与接口连接。

## 三个执行入口

先同步源码清单，在服务器code目录执行专项检查。

```bash
PYTHONPATH=src python -m unittest discover -s tests -p test_mstd_temporal.py -v
python tools/run_mstd_temporal_controls.py --backend torch --device cuda --plan-only
python tools/run_mstd_temporal_controls.py --backend torch --device cuda
```

理想协议的六个条件和六种动力学共36条轨迹。结果写入带北京时间的experiments/validation运行目录，含plan.json、summary.json、各设置jsonl及原生输入。

已有MT录制可用帧采样重放进行诊断。

```bash
python tools/run_mstd_temporal_controls.py --backend torch --device cuda \
  --recording experiments/validation/<父实验>/<条件>/baseline
```

重放采用mt_readout_activity的实际帧末可用时间，并从该时刻保持一个采样间隔，新增的保持延迟在记录中明确标注。原神经毫秒级轨迹需要通过端到端入口重新运行。缺少相机内参时记录群体活动和几何接受，航向角保持空值。

固定MT的端到端入口使用原图像DAT，每个MT分支通过同一V1重放进行六种动力学比较。两种MT分支乘六种动力学，共12次运行。

```bash
python tools/run_mstd_temporal_end_to_end.py \
  --config configs/model_python_spiking_baseline32_v01.json \
  --stimulus <有自运动真值的现有DAT路径> --frame-ms <实际帧间隔ms> \
  --intrinsics <与输入尺寸匹配的内参JSON> --backend torch --device cuda
```

该入口每个神经tick正常推进MT与MSTd，并核对同一MT分支的读出哈希。输出保留完整模型配置、监视器和MSTd逐帧估计。端到端航向评价需结合对应序列真值。

生成理想协议的航向曲线MP4。

```bash
python tools/render_mstd_temporal.py --run experiments/validation/<本次理想协议运行ID> --case step
```

金色为真值，青色为正式接受的航向，拒绝期间以断线表示。输出位于outputs/comparisons/mstd_temporal，六个面板采用共同角度范围和时间轴。

## 同步清单

```text
src/mstd/model.py
src/mstd/mstd.py
src/mstd/temporal.py
tools/run_mstd_temporal_controls.py
tools/run_mstd_temporal_end_to_end.py
tools/render_mstd_temporal.py
tests/test_mstd_temporal.py
configs/model_python_spiking_temporal_baseline32_v01.json
configs/model_python_spiking_temporal_spatial32_v01.json
docs/MSTD_TEMPORAL_STAGE3_V01.md
```

新增配置模板可直接用于run_model.py。两个MT分支均采用32神经积分子步。当前参数与入口用于继续验收时间竞争，后续根据误差、有效率、抖动、切换与恢复指标选择配置。
