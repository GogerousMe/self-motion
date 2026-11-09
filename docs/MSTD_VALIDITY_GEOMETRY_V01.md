# MSTd几何有效性对照v01：操作与验收

日期：2026-11-09。目标是检验当前纯平移径向扩张解码器的有效性判据。保留原有规则；本轮不修改V1滤波、脉冲采样、MT神经元、模板或竞争动力学。

## 1 本轮已经完成什么

- 新增 `src/mstd/validity.py`，两种模式：默认 `legacy_margin`；显式配置后启用 `geometry_support`。
- 列表参考实现、NumPy实现、torch实现共用同一个标量决策函数。CUDA上仅在导出读出时计算额外摘要，不把整幅响应搬回CPU。
- 增加独立配置 `configs/model_python_spiking_geometry_v01.json`，旧配置保持原样。
- 15个数值开发条件通过后，将参数写入冻结文件，再生成并测试30个数值留出条件，全部达到本协议目标。留出航向为-17、-12、-7、-2.5、2.5、12、17度；噪声/随机背景种子为303、404、505，开发种子为101。
- 对已有两段CUDA响应直接重新判断有效性：静止误报35%→0%；前进有效率40%→100%。前进未门控竞争峰值MAE维持0.417°。旧规则只接受完全正确的一部分帧，新规则接受更多帧，因此有效帧MAE从0变为0.417°；这不是航向峰值被修改。
- 54项相关测试运行成功：48项通过，6项因本机缺少torch而跳过。本机没有验证torch/CUDA新分支；服务器验证仍需执行。

数值验证目录：

```text
experiments/validation/20261109__mstd_geometry_gate_80x80_f000080__py_rate_cpu__r01/
```

最终检查脚本对已有响应的复核：

```text
experiments/validation/20261109__mstd_geometry_saved_check_80x80_f000080__py_rate_cpu__r01/
```

已有真实场景此前用于定位问题，不能当成盲测。数值留出通过也不代表完整图像链路通过。

## 2 新判据具体判断什么

令每个位置的MT向量为v，长度为a，当前竞争赢家的单位径向模板为t。维持现有候选航向-20°～20°、间隔5°；没有新增插值解码器。

| 条件 | 冻结数值 | 含义 |
|---|---:|---|
| 平均MT向量长度 | ≥0.02 | 输入强度不能过弱，沿用DynamicsConfig.min_evidence |
| 有效位置占比 | ≥25% | a≥0.02的位置应覆盖足够多的像素，防止极少数强点决定全局结果 |
| 当前赢家的径向匹配分数 | ≥0.20 | S=max(0, Σ(t·v)/Σa)，检查当前响应是否符合扩张几何 |
| 相对统一平移场的几何优势 | ≥0.10 | S−U；U=长度(Σv)/Σa，是最优单一方向场的支持度 |

四条同时满足才接受。零输入的S、U均设0。这里的强度、分数和占比均无量纲，既不是实际放电率Hz，也不是已校准概率。

几何分数检查的是当前竞争赢家，而非另外挑一个瞬时峰来替它背书。`margin`依然输出并用于旧规则，但不再作为新规则的必需条件。图像统一向右运动的局部方向可以是正确的；它在当前有限视野内不提供足够的径向扩张证据，因此本MSTd基线拒绝输出航向。

新规则针对当前视野内、中心附近的扩张模板。后退收缩、旋转、弯道以及扩张中心在视野外的运动不属于此次成功解码范围；也没有得到速度、尺度、旋转或SE(3)位姿。若MT静态误响应本身恰好形成径向结构，新规则仍可能误报；MT静态残留还需要后续单独处理。

## 3 冻结文件与独立图像刺激

权威冻结参数：

```text
experiments/validation/20261109__mstd_geometry_gate_80x80_f000080__py_rate_cpu__r01/frozen_validity.json
```

参数使用此前诊断支持的预设值，经开发条件确认后冻结，没有在新留出响应上搜索阈值。不要改动这个文件。后续若改判据或阈值，应创建v02并重新划分开发与留出集。

已在冻结后生成新的图像套件：

```text
input/stimuli/selfmotion/mstd_gate_holdout_s0097__80x80__f000080__dt50ms__7cases__v01/
```

| DAT | 验证内容 | 期望 |
|---|---|---|
| cloud_static.dat | 同种云点纹理保持静止 | 稳态拒绝航向，检查误报 |
| cloud_forward.dat | 云点前进0° | 稳态接受，峰值接近0° |
| cloud_heading_left7.dat | 云点纯平移-7°，不在模板中心 | 接受，最近候选-5°，允许量化误差 |
| cloud_heading_right7.dat | 云点纯平移+7° | 接受，最近候选+5° |
| ground_forward.dat | 地面分布前进0° | 检查空间分布改变后的泛化 |
| ground_heading_left10.dat | 地面纯平移-10° | 接受，峰值接近-10° |
| ground_heading_right10.dat | 地面纯平移+10° | 接受，峰值接近+10° |

场景种子97，旧套件为0；建议神经脉冲种子43，旧实验为42。每段80×80、80帧、50ms/帧，4秒；各条件独立运行、重置状态，不把段间残留引入此次比较。同一场景类型各条件共用场景种子以控制初始点分布。

DAT沿用25字节VisualStimulus头与单通道灰度像素；相机内参、真值、manifest、SHA256和登记均已配齐。点深度仅在真值中，V1读取的仍是普通图像。`preview/*.mp4`为H264、20fps、80帧，已用ffprobe核对。不要把预览视频或真值当模型输入。

## 4 需要同步到服务器的内容

可以继续直接复制源码，不需要打zip。同步以下新增/改动文件及完整的新刺激目录：

```text
src/mstd/validity.py
src/mstd/mstd.py
src/mstd/tensor_pipeline.py
src/mstd/model.py
src/mstd/cli.py
configs/model_python_spiking_geometry_v01.json
examples/check_mstd_validity.py
tools/run_selfmotion_baseline.py
tools/prepare_mstd_gate_holdout.py
tests/test_mstd_validity.py
docs/MSTD_VALIDITY_GEOMETRY_V01.md
input/stimuli/selfmotion/mstd_gate_holdout_s0097__80x80__f000080__dt50ms__7cases__v01/
experiments/validation/20261109__mstd_geometry_gate_80x80_f000080__py_rate_cpu__r01/frozen_validity.json
```

同时将 `input/registry.csv` 中本套件的7行同步到服务器索引。服务器运行工具会自行追加实验登记；服务器已有的 `experiments/registry.csv` 保留并合并新增记录，避免用本机索引覆盖服务器的新运行。

环境沿用已有mstd conda环境与CUDA PyTorch，无需新增Python依赖。运行前进入实际代码根目录，例如 `/data/workspace/WYQ/mstd/code`。

## 5 服务器实际执行顺序

### 第一步：设置位置，确认torch和设备

```bash
conda activate mstd
cd /data/workspace/WYQ/mstd/code
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -m unittest discover -s tests -p test_mstd_validity.py -v
```

CUDA必须为True。新增测试会在torch可用时同时核对torch CPU和NumPy/list的判据语义；真正CUDA链路由后续四段实验验证。

### 第二步：先跑四段云点条件

```bash
SUITE=input/stimuli/selfmotion/mstd_gate_holdout_s0097__80x80__f000080__dt50ms__7cases__v01
FREEZE=experiments/validation/20261109__mstd_geometry_gate_80x80_f000080__py_rate_cpu__r01/frozen_validity.json

for CASE in cloud_static cloud_forward cloud_heading_left7 cloud_heading_right7; do
  python tools/run_selfmotion_baseline.py \
    --case "$CASE" \
    --suite "$SUITE" \
    --config configs/model_python_spiking_geometry_v01.json \
    --frozen-config "$FREEZE" \
    --seed 43 --backend torch --device cuda || break
done
```

工具会检查刺激索引、哈希、分辨率、帧数和冻结参数一致性。每段自动生成北京时间运行ID，输出例如：

```text
experiments/validation/<time>__cloud_forward_80x80_f000080__py_spike_cuda__geometry_support_s0043__r01/
```

原生输出含 `model_config.json`、`model_report.json`、`estimates.jsonl`、`monitors/` 和 `artifact.json`。`estimates.jsonl` 中 `validity.legacy_valid` 是同一响应下旧规则的判断；`valid`/`bearing_valid` 是新规则判断。模板分数、竞争峰值和神经响应不会因两个规则而重新计算为两套。

每段完成后，工具会自动生成旁边的 `<run_id>__gate_check/REPORT.md`，并打印 `GATE_REPORT:`。这个检查直接使用原始分数和赢家，同时从保存MT重算空间摘要，逐帧核对CUDA原生判断与CPU重算；不使用帧末近似重放。

如果某次报错，先查看报错并处理，再继续未完成条件。失败运行仍保留登记。`completed`表示执行完成；是否满足诊断目标见报告。

### 第三步：再跑三个地面条件

在同一终端保留SUITE和FREEZE变量：

```bash
for CASE in ground_forward ground_heading_left10 ground_heading_right10; do
  python tools/run_selfmotion_baseline.py \
    --case "$CASE" \
    --suite "$SUITE" \
    --config configs/model_python_spiking_geometry_v01.json \
    --frozen-config "$FREEZE" \
    --seed 43 --backend torch --device cuda || break
done
```

## 6 检查什么，怎样判定

统一排除前20帧滤波/网络预热，主统计后60帧。逐帧结果仍完整保留；本轮不把预热排除等同于已验证起步和切换延迟。

1. 静止：新规则有效率即误报率，目标≤5%。同时查看被哪条规则拒绝，确认不是输入被错误读为零。
2. 正向运动：新规则有效率目标≥95%。同时检查未门控赢家MAE≤2.5°、有效帧MAE与最大误差，不能只用筛选后的误差证明成功。
3. ±7°：期望最近候选为±5°，约2°量化误差正常。有效率改善不代表已经实现连续角度估计。
4. 地面：在空间分布改变后仍应满足上述正例目标。若失败，保留失败结果，不用这几个留出条件继续调v01阈值；先分析证据、几何优势、活动覆盖率和未门控方向。
5. 后端：自动检查要求原生CUDA新判据与保存响应的CPU标量重算逐帧一致。若不一致，优先处理实现/元数据问题。

报告中主要指标为 `legacy_valid_fraction`、`geometry_valid_fraction`、`unconditional_winner_mae_deg`、`geometry_valid_mae_deg`。`frames.csv` 还包含 `evidence`、`winner_template_score`、`uniform_score`、`geometry_excess`、`active_fraction`、`rejected_by`。单帧不是独立试验，这里的比例仅代表这些短序列，不能据此宣称已验证真实车载SLAM。

也可以将多个运行合并分析，以下ID需要替换成终端实际打印的RUN_ID：

```bash
python examples/check_mstd_validity.py \
  --frozen-config "$FREEZE" \
  --parent-run 实际静止运行ID \
  --parent-run 实际前进运行ID \
  --output experiments/validation/manual__mstd_gate_independent_group__py_rate_cpu__r01
```

该命令仅分析保存响应，不重跑V1/MT。输出目录必须未存在；再次执行使用新的r编号。当前脚本仅处理本协议的80帧、80×80、50ms、9模板且具有内参的运行。

## 7 何时转入MT问题处理

先完成四段云点，随后检查三个地面条件。若新判据确实降低误报并保留正确航向，再固定它作为对照，开始MT专项：静态纹理残留、八方向峰值与向量方向差异、连续切换残留。届时在V1不变的条件下逐项检查MT兴奋/抑制与读出，保留本轮配置作基线。

新MSTd判据只是拒绝不可靠航向，并没有消除MT静止放电；不能以有效率改善作为MT问题已经解决的证据。若本轮独立图像输入尚不能产生正确的未门控航向，应先沿V1投影→MT方向场→MSTd模板逐层定位，再决定改哪一层。
