# 2026-11-08 状态回顾与下一轮测试

本次为源码、现有原生报告和输入协议审查，没有新运行神经仿真，没有重生成输入，也没有覆盖已有实验。遵守AGENTS.md及FILE_ORGANIZATION.md。

## 当前主线与此前的区别

主配置为 configs/model_python_spiking.json，统一入口run_model.py：

二维灰度/RGB图像DAT → Python MotionEnergy V1连续响应[3,28,H,W] → 每1ms本地泊松计数 → 延迟/快速E/I电导 → Izhikevich MT-E/MT-I[8,H,W] → 因果活动迹/方向读出 → 连续MSTd适配器。

- V1是有历史状态的固定时空运动能量前端；并未变成膜电位神经元。原CUDA库数值等价性仍需独立实测。
- MT已是真正维护v/u、阈值复位、突触电导、延迟、脉冲事件的Python网络，可以统计实际MT-E/I发放率。
- mt_radius=0是该后端要求，没有旧连续版3×3空间池化；filtered rate是时间滤波，不是空间池化。
- 正负权重分别进入E/I通路，不等价于旧连续式“有符号求和后整流”。旧版通过不能自动证明新版通过。
- 神经时间步默认1ms，积分4个子步。V1每图像帧更新一次，其率在帧内驱动泊松输入；MT/MSTd在细时间步更新。
- MSTd仍是连续状态的径向模板/竞争适配器，不是正式脉冲MSTd；反馈接口存在但默认关闭。
- 新原始.events为tick、neuron、count记录，不是CARLsim SpikeMonitor DAT，不能用旧inspect-spikes解码。
- 三个源兼容V1尺度块共享基础滤波结果，不能仅凭三块形状宣称已完成独立多尺度/速度标定。

## 已有证据与未完成内容

本次直接读取 experiments/validation/legacy__grating_32x32_f000040__py_spike_cpu__r03/validation.json，确认7项受控机制筛查通过、八方向均有响应且峰值方向正确。该结果为NumPy CPU、32×32、40帧/方向的特定条件，不是80×80长序列、多种子或GPU验收。

文档PYTHON_SPIKING_VALIDATION.md记录当时79项回归、77通过/2项缺PyTorch跳过。本次终端默认Python无NumPy，未重跑完整回归、未安装依赖，不将历史测试数量当作本次执行。

PYTHON_V1_VALIDATION.md记录旧连续路径的随机纹理存在峰值偏向邻近方向而群体向量接近真值的现象。它不是新脉冲MT的结论，但必须在下一轮同时报告峰值和群体向量，不允许仅换判据宣称问题消失。

纹理、静止/停止、无复位切换和自运动基线尚未在新主线上完成任务验收。自然图像的展示也不能替代带运动真值的验收。

## 两份已有刺激：已审计，可复用

| 内容 | 当前路径 | 帧数/时长 |
|---|---|---|
| 纹理移动、静态、连续切换 | input/stimuli/texture/texture_sequence__80x80__f001800__dt50ms__v01/all_stimuli.dat | 1800帧/90s |
| 五条件自运动 | input/stimuli/selfmotion/selfmotion_sequence__80x80__f000400__dt50ms__v01/all_stimuli.dat | 400帧/20s |

均为80×80单通道、50ms/帧。实测文件头type=4；源文件保持原始协议，不为了新图片转换器默认type=5而改写。当前加载器支持既有类别，不能只按.dat扩展名分类。

本次核对DAT长度、头部尺寸/帧数、manifest、artifact.json、input/registry.csv哈希一致；CSV连续覆盖27个纹理段/5个自运动段。自运动truth.jsonl有400行并保留条件局部坐标，intrinsics.json存在。旧outputs/validation_sequences_01不是新运行的权威路径，迁移索引保留历史映射。

## 测试顺序

### A. 新主线运行环境与80×80短链路

固定python_spiking配置、seed=42、dt_ms=1、substeps=4、同一投影。先在目标运行环境执行现有神经/监视器回归，CPU/GPU分别记录。GPU随机源不要求逐事件等于CPU；比较相同语义、有限状态、统计及任务结果。

对纹理序列先做少量帧性能/记录试跑，再处理前160帧，覆盖初始80帧静止及80帧右移。注意：仅运行前40帧全是静止，不能声称验证了运动方向。检查运行时长、实际文件体积、有限电压、事件到帧率回读、通道坐标和时间戳。

保留源码/投影/实际配置/输入哈希；估算完整1800帧的计算和磁盘占用后再正式运行。不要为加速改变仿真时间或监视器语义而不记录。

### B. 完整纹理方向与神经响应测试

一次连续处理1800帧，V1与MT在整个序列内保持状态。按27段分别评价：

- MT-E/I实际Hz的八方向调谐、群体向量圆周误差、有效覆盖率；分清逐帧计数率和因果滤波率。
- 0～4s静止、运动停止后保持纹理、最终静止的误报和消退。启动瞬态单列，不把零向量强行赋0°。
- 54～86s无黑场连续切换的响应延迟、错误方向持续时间和残留。稳定方向误差与切换过程分别评价。
- 空间图、原始事件栅格、E/I电压采样与速率；帧末电压不冒充完整毫秒电压波形。
- 先固定种子定位问题，再以独立种子重复，区分场景随机性与脉冲随机性。正式阈值先在开发条件确定，再冻结测试。

旧analyze_mt_gratings.py是原V1 SpikeMonitor投影链路，不作为新Python事件格式的读取入口。应使用统一模型原生monitors/events和visualize_python_spiking.py，并补齐分段任务汇总。当前rate-max仅为画图，不修改真实率。

### C. 尽早并行检查MSTd几何，再做真实图像纯平移

理想方向场可独立验证中心、左右偏移、静默和MSTd竞争，不依赖先完成所有MT扩展。

真实图像采用五条件：cloud_static、cloud_forward、ground_forward、ground_heading_left10、ground_heading_right10。对应目标为无有效航向、0°、0°、-10°、+10°。提供配套真实内参，默认配置intrinsics为空时只有像素模板，不能报告标定角度。

建议第一轮先用保留的5个原独立DAT，各从零状态开始运行完整80帧，避免拼接场景切换混入最初基线。原套件位于input/stimuli/selfmotion/selfmotion_suite__80x80__f000080__dt50ms__14cases__v01/，5个文件短名保持不变。

若使用合并400帧DAT，必须有分段评估和适当的MSTd状态处理；当前simulate()只遍历图像，不自动读取分段CSV并重置MSTd。因此不能一条默认命令跑完就称为五个独立条件验收。也不能为了重置MSTd连带清掉本应连续的V1/MT历史而不说明实验定义。

输出模板分数、9个候选活动随时间、航向误差/偏差/P95、覆盖率、静止误报、启动延迟和边界模板占用。非零输出不等于估计正确。

### D. 基线后扩展

加入网格外航向、独立场景/纹理种子及速度变化；再比较瞬时匹配、普通平滑、时间竞争。在动态物体和遮挡下同时测误差与跟踪延迟。

正式脉冲MSTd作为独立后端，保留当前连续适配器作为对照，不与V1/MT重调混在一起。收缩/旋转/曲率随后逐项增加；完整SLAM位姿不是本轮门槛。

## 后续刺激与产物必须遵守的规则

- 优先复用已登记的DAT，不因模型改变重复生成同样像素的数据。
- 新刺激写入input/stimuli/<category>/<规范ID>/；独立DAT使用完整ID及.dat.json，协议套件只规范外层目录，内部manifest引用不拆散。
- 像素、帧序、时长或真值改变必须新版本，记录几何、时间、方向、哈希和源数据；不得覆盖原始文件。
- 创建前检查input/registry.csv和experiments/registry.csv；产物加artifact.json并登记父运行/刺激ID。
- 试跑进experiments/scratch，专项验证进experiments/validation；正式响应进outputs/recordings，响应图/MP4进outputs/visualizations，纯输入预览随刺激包。
- 部分旧生成/仿真入口只保存原生metadata，不自动完成registry/artifact登记；使用入口后仍须按规范补齐，不能假定自动合规。长期应统一登记入口，避免手工遗漏。
- 监视器保留实际单位、帧号及tick时间。改变色标生成新view版本，不复制或改写原始响应。

本轮不生成新刺激、不创建伪passed实验、不更改已有源码或配置；下一步优先A/B，C的理想后端检查可以同时推进。
