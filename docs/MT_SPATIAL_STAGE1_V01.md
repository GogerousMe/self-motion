# MT空间整合第一阶段

日期为2026-11-16。当前完成空间回路开发、机制检查、CPU小型对照及MP4展示。GPU完整八方向与多速度验收待服务器运行。

## 1 实现内容

每个位置保留8个方向的E/I神经元。新增同方向MT-E到邻域MT-E的空间兴奋连接，脉冲在tick末写入延迟线，在之后的tick形成独立指数衰减电导并参与膜电位更新。

空间核为沿优选方向的椭圆高斯，包含中心位置。图像边界采用有效邻域权重归一化。方向相差180°的核具有相同空间形状，八个通道分别计算。

第一版参数为半径2像素、沿方向标准差1.5像素、垂直标准差0.75像素、增益0.05、时间常数5ms、延迟1个1ms tick。神经元积分采用32子步。上述空间参数为工程初始选择，尚未经过完整功能调参。

新增配置字段为mt_spatial。原mt_radius继续描述原有率模型的池化设置，脉冲模型使用mt_radius=0和独立mt_spatial配置。

## 2 已执行的检查

- 九项空间专项检查通过，覆盖常量场、旋转、镜像、核轴向、局部性、通道隔离、配置、延迟、重置、更新顺序和正常模型入口。
- 相关回归共执行51项测试并通过。CUDA测试类因本机缺少CUDA设备跳过。
- NumPy与PyTorch CPU的空间核及固定事件神经回路对照通过。
- 关闭空间回路时，与修改前源码运行同一100-tick输入所得的脉冲、膜电位、恢复变量及率迹摘要逐位一致。
- 90°旋转检查的最大绝对差约为2.38×10^-7。矩形网格的90°、180°、270°旋转及镜像检查均通过。
- 常量场检查覆盖边缘、角点、1×1网格及核半径大于图像尺寸的情况。
- 45°方向检查核的轴向与对向几何关系。完整图像旋转45°涉及方格网格插值，其误差需单独评价。
- 已通过run_model.py的PyTorch CPU图像入口，模型正常保存V1、MT、MSTd和空间电导监视器。

小型对照运行位于experiments/validation/20261116_195550__mt_spatial_controls__py_spike_cpu__sub32__s0043__r01。mechanism_checks.json与mechanism_tests.log保存机制及回归结果，source_snapshot保存该小型运行记录哈希对应的源码。

## 3 小型对照结果

两个条件均为32×32、24帧、每帧50ms。前20帧预热，评价帧为20至23，共4帧。随机纹理种子0，脉冲种子43。

baseline从图像计算V1并录制。spatial_only重放baseline的同一V1率数组，使用相同Poisson种子与积分设置。

| 指标 | baseline | spatial_only |
|---|---:|---:|
| 静态纹理MT平均放电率 | 5.318Hz | 5.398Hz |
| 静态MSTd几何接受帧 | 0/4 | 0/4 |
| 向右纹理运动整体方向MAE | 1.821° | 1.681° |
| 向右纹理运动活跃像素方向MAE | 44.018° | 43.788° |
| 向右纹理运动MT平均放电率 | 19.983Hz | 20.293Hz |

局部方向评价裁去6像素边界，并使用合成向量幅度≥0.02的像素。整体方向由区域向量均值计算。覆盖率和可评价帧数在comparison.json中一起报告。

空间通路已对实际放电产生影响。该小样本中方向误差略有降低，静态放电率同时增加约1.5%。局部方向误差仍较大。完整方向、速度、噪声和独立输入验收仍待完成。

追加的16子步对照中，整体方向MAE为baseline 1.666°、spatial_only 1.700°。32子步下的小幅收益尚未在该积分步长对照中复现。两个步长的平均率变化低于0.5%，方向精度收益仍需要完整刺激与独立种子确认。

这些纹理夹具使用像素模板，未设置相机内参。其MSTd valid字段用于几何接受检查，未产生标定航向角。

## 4 监视器与数值范围

mt_exc_rates保存每帧实际兴奋性脉冲计数除以帧时长，单位Hz。mt_readout_activity保存无量纲方向读出。新增mt_spatial_exc_conductance保存空间通路的有效电导。mt_exc_conductance包含输入、反馈接口及空间通路的总兴奋电导。

神经积分器在每个积分子步检查有限性。frame_end_ranges.json记录已保存的帧末电压、电导范围。帧末范围的采样时间在文件中明确标记。

本机PyTorch版本为2.15.0+cpu。CUDA卷积、GPU时间开销及完整刺激的数值收敛需要在服务器检查。机制参数验收继续使用32子步参考，并保留16子步对照结果。

## 5 同步到服务器的文件

从项目code目录同步以下改动。

```text
src/mstd/mt_circuit.py
src/mstd/spiking_mt.py
src/mstd/model.py
src/mstd/monitors.py
tests/test_mt_spatial.py
configs/model_python_spiking_baseline32_v01.json
configs/model_python_spiking_spatial32_v01.json
tools/run_mt_spatial_controls.py
tools/render_mt_spatial_controls.py
docs/MT_SPATIAL_STAGE1_V01.md
```

运行工具还会复用已有src/mstd、configs/v1_projection.json与examples中的读出、录制和展示代码。直接同步当前源码目录可保持这些依赖一致。

## 6 服务器执行顺序

在服务器的code目录执行专项检查。该检查在检测到CUDA时会包含固定事件CUDA对照。

```bash
PYTHONPATH=src python -m unittest discover -s tests -p test_mt_spatial.py -v
```

先查看小型GPU对照计划，再运行。工具自动创建全新运行ID和对应夹具，登记artifact及registry。

```bash
python tools/run_mt_spatial_controls.py \
  --backend torch --device cuda --quick-check --plan-only

python tools/run_mt_spatial_controls.py \
  --backend torch --device cuda --quick-check
```

小型GPU检查完成后运行静态纹理与八方向×四速度矩阵，共33个条件，每条件baseline/spatial_only两次运行。

```bash
python tools/run_mt_spatial_controls.py \
  --backend torch --device cuda \
  --cases static,texture --substeps 32 --seed 43 --gain 0.05 --radius 2
```

需要检查积分步长时对同一矩阵追加--substeps 16，工具生成另一运行目录。

原KITTI动态与冻结图像可单独对照。该命令需要已有两个父运行的完整V1录制，动态条件还需要匹配的刺激元数据和真值。

```bash
python tools/run_mt_spatial_controls.py \
  --backend torch --device cuda --cases kitti --plan-only

python tools/run_mt_spatial_controls.py \
  --backend torch --device cuda --cases kitti
```

单次正常模型运行也可使用新增spatial32配置和原run_model.py入口，frame_ms需与所选刺激一致。baseline32配置用于相同输入的关闭空间通路对照。

## 7 输出与展示

每次运行自动保存到experiments/validation/<北京时间>__mt_spatial_controls__py_spike_cuda__sub32__s0043__r01。

```text
plan.json
artifact.json
summary.json
static_texture/
  stimulus.dat
  stimulus.dat.json
  case.json
  comparison.json
  baseline/
  spatial_only/
texture_d000_s1/
  ...
```

baseline与spatial_only目录各自保存model_config.json、model_report.json、estimates.jsonl、frames.csv和原生monitors。基线保存完整V1，空间候选通过重放该V1进行配对。

对静态或纹理条件生成共享色标MP4时，将RUN替换为本次打印的运行目录。

```bash
RUN=experiments/validation/<本次运行ID>
python tools/render_mt_spatial_controls.py --run "$RUN" --case texture_d000_s1
python tools/render_mt_spatial_controls.py --run "$RUN" --case static_texture
```

MP4位于outputs/comparisons/mt_spatial/<运行ID>__<条件>__rates__v01/mt_spatial_comparison.mp4。上排为baseline，下排为spatial_only，八方向固定顺序为右、右下、下、左下、左、左上、上、右上，色标单位Hz。

本阶段完整验收重点为各方向/速度的局部误差、整体误差、覆盖率与静态响应共同变化。GPU矩阵完成后据此选择空间参数，再进入方向竞争开发。
