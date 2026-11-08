# Python V1–MT：默认GPU运行与CPU切换（2026-11-08）

## 实施范围

保留同一套Python神经模型，使用PyTorch CUDA张量承载V1滤波、泊松采样、MT的v/u状态、E/I电导、延迟队列和脉冲迹。固定连接仍由张量投影表达。没有增加CARLsim依赖或C++构建步骤。

MSTd当前仍是连续率适配器，在同一设备上逐神经时间步推进。后续可用脉冲MSTd替换；MT输出仍提供设备上的exc/inh事件，延迟反馈接口保留。GPU迁移不代表MSTd已经脉冲化，也不构成与CARLsim数值等价的证据。

## 默认规则

| 入口或参数 | 实际计算 |
|---|---|
| `python run_model.py ...`，未指定配置 | `model_python_spiking.json`，PyTorch/CUDA |
| Python脉冲配置未写backend/device | PyTorch/CUDA |
| `--device cpu` | NumPy/CPU |
| `--backend numpy` | NumPy/CPU |
| `--backend torch --device cpu` | PyTorch/CPU，用于后端比较 |
| `--device cuda:1` | PyTorch，第二块CUDA设备（统一入口支持） |
| 显式`--config configs/model.json` | 原连续率基线，保留其CPU配置 |

CLI设备参数优先于配置。没有安装PyTorch或CUDA不可用时明确报错，需修复环境或显式指定`--device cpu`。模型不会静默改用CPU。自然序列工具与机制验证脚本采用同一默认规则。

## 服务器环境与运行

在仓库code根目录、现有Python环境执行。已安装可用CUDA版PyTorch时不必重装；基础安装`pip install -e .`仅依赖NumPy，GPU需要额外具备CUDA版PyTorch。

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -c "import torch; print('torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
python -m unittest discover -s tests -p test_gpu_execution.py -v
python -m unittest discover -s tests -p test_python_v1.py -v
```

`test_gpu_execution.py`包含执行策略、CPU命令行端到端、固定事件CPU/GPU神经元对照、CUDA完整图像链路、反馈/观察器、溢出检查。缺少PyTorch/CUDA时CUDA测试显示skipped；skipped不是GPU通过。

自然刺激建议先用Daria的10帧试跑，工具会校验输入哈希、生成规范ID、登记artifact/registry并制作视频：

```bash
# 默认GPU；--dry-run可仅检查计划
python tools/run_natural_spiking.py --sequence daria_walk --stage scratch --max-frames 10 --repeat 1

# 同一输入、帧数和监视器，CPU对照
python tools/run_natural_spiking.py --sequence daria_walk --stage scratch --max-frames 10 --repeat 1 --device cpu

# 验证完成后，GPU处理全帧并保存MT响应及视频
python tools/run_natural_spiking.py --sequence daria_walk --stage recording --repeat 1
python tools/run_natural_spiking.py --sequence paragliding --stage recording --repeat 1
python tools/run_natural_spiking.py --sequence snowboard --stage recording --repeat 1
```

运行ID自动区分`py_spike_cuda`/`py_spike_cpu`，时间、分辨率、帧数和重复编号一并保留。路径和视频文件见[NATURAL_SPIKING_RUNS.md](NATURAL_SPIKING_RUNS.md)。重复运行如发生ID冲突请增加repeat，不覆盖已有目录。

直接使用现有DAT的通用入口如下，输入路径替换为服务器实际位置。模板max_frames为null，省略`--max-frames`会处理全部帧：

```bash
python run_model.py --stimulus /实际路径/刺激.dat --max-frames 10 \
  --output experiments/scratch/manual__v1mt_gpu_preview__r01
python run_model.py --stimulus /实际路径/刺激.dat --max-frames 10 --device cpu \
  --output experiments/scratch/manual__v1mt_cpu_preview__r01
```

通用入口输出原生记录；单独运行时仍须按FILE_ORGANIZATION规范补充输入、运行、展示的artifact和registry。自然序列工具会自动完成这些登记。

## 实现与性能指标

神经步长、子步、投影权重、时间常数、阈值复位和因果延迟保持同一配置。GPU帧内数值检查留在设备上汇总，帧末一次判断，保留复位前溢出标志；外部V1率在采样前立即检查。异常不回滚神经状态，应终止该次实验或reset。

未注册反馈控制器时，MT evidence和MSTd报告到帧末才转为Python数值，MSTd动力学仍每步推进。注册现有控制器后恢复每步状态导出。启用原始脉冲监视器仍会逐步复制事件至CPU写文件；视频制作在仿真后进行。这些操作可能限制整体加速。

验收时使用相同输入哈希、尺寸、帧数、dt、seed、监视器和内参，分别保存CPU/GPU完整结果：

| 项目 | 判定或记录 |
|---|---|
| 设备 | `model_report.json`中model.backend=torch、device=cuda或cuda:N；execution记录PyTorch/CUDA版本与显卡名 |
| 完整性 | complete=true、frames与计划一致，输入SHA256一致；可视化能读取原生监视器 |
| 机制 | CUDA测试实际执行；固定事件E/I脉冲计数逐步一致，膜电位atol=1e-3、rtol=1e-5，抑制电导atol/rtol=1e-5 |
| 前端 | test_python_v1中的CPU/CUDA原始率对照atol/rtol=5e-4 |
| 随机网络 | 同设备同种子可复现；NumPy和CUDA泊松算法不同，不要求跨设备事件逐个相同。八方向需有响应，峰值方向误差≤22.5°；自然视频无真值时不报告航向准确率 |
| 构建耗时 | initialization_seconds，已包含CUDA初始化同步 |
| 仿真耗时 | compute_seconds，包含帧前后CUDA同步；启用逐步脉冲监视器时也包含其中的写出 |
| 运行耗时 | wall_seconds / wall_fps，含输入读取和监视器写出，不含初始化、输入预检哈希和事后视频 |
| 加速比 | 同条件CPU耗时/GPU耗时，分别列compute和wall；至少重复3次报告中位数 |

显卡利用率与显存峰值可在另一终端观察`nvidia-smi`。大型854×480输入不要从小尺寸测试推算性能或显存；完整V1与脉冲记录的磁盘开销不因GPU而减少。

## 本轮验证状态

本机没有PyTorch/CUDA。本轮完整回归共97项，91项通过，6项PyTorch/CUDA相关测试跳过，包含真实CPU命令行入口及原生记录检查。CUDA测试已提供，尚未在目标服务器执行。因此默认设备已改为GPU，但实际GPU兼容性、方向筛查和加速比仍待服务器证据，不宣称已提速。
