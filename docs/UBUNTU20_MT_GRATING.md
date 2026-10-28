# Ubuntu 20.04：连续八方向光栅的 V1 DAT 解码与 MT 测试

日期：2026-10-25。采用源码目录复制与增量同步部署，不要求打包 ZIP。保留本机目录结构，以下命令在服务器 `MSTD/code/` 中执行，均为 Bash 命令。

## 1. 本轮测试链路与代码检查结论

输入光栅 DAT → 服务器现有 C++ CARLsim/V1 → spk_V1.dat → 按时间窗计数 → 28→8 带符号投影及三尺度汇总 → 实际 Python MT 池化／归一化 → 逐帧方向和分段报告。

两种 DAT 格式不同：输入是 VisualStimulus 图像帧，输出是 SpikeMonitor 脉冲事件；不要混用。Python 不直接运行服务器 V1，也不复现 CARLsim MT 的脉冲细胞动力学。

已核查：

- 项目要求 Python >=3.10；Ubuntu 20.04 默认 Python 3.8 不满足要求，不替换系统 Python。
- Python 解码与 MT 分析只有标准库依赖，无需 NumPy、PyTorch、CUDA、MATLAB 或 ffmpeg。
- `inspect-spikes` 检查格式；默认只抽查前 10000 事件，不能替代全文件验证。
- 旧 `analyze-calibration` 适用于八个独立输入文件的 v1 清单，统计投影调谐，不调用实际 MT。
- 新 `examples/analyze_mt_gratings.py` 支持当前连续 v2 清单，按 CSV 分段，并实际调用 `encode_mt`。
- `mstd run` 是 MSTd 膨胀航向推断，本轮不用它作为光栅方向测试入口。

## 2. 上传哪些文件

直接将本机 `D:\A_Gorger\program\MSTD\code` 复制到服务器固定路径，例如 `/home/你的用户名/projects/MSTD/code`。也可以复制外层 MSTD，保留其中 code 子目录。后续只同步修改的源码文件，不需要每次打包。建议保留以下布局：

```text
MSTD/
  code/
    src/mstd/                         # 整个 Python 包
    configs/v1_projection.json        # 原 C++ 三组方向投影
    examples/                        # 包含分析及测试所需采集脚本
    tests/
    pyproject.toml
    README.md
    THIRD_PARTY_NOTICES.md
    docs/
    srcipts/bw_grating_output/sequence_20261019_134626_053/
      all_stimuli.dat
      all_stimuli_segments.csv
      manifest.json
```

最小运行集为 `src/mstd/`、投影 JSON、分析脚本和刺激目录三个文件。首次建议同步 src/、configs/、examples/、tests/、docs/ 和项目说明文件，并单独同步本次使用的刺激目录。`srcipts` 是当前项目实际目录名，保留原拼写。

不需要上传 `.git/`、`.venv/`、`__pycache__/`、`MathWorks/`、MATLAB 用户偏好／证书、既往 outputs/、全部参考 PDF。已有输入 DAT 时不需要上传 MATLAB 生成器。

服务器原有 CARLsim/CUDA/修改版 MotionEnergy 工程继续使用。该工程及其库不包含在当前 Python 项目中；仅复制本目录不能从零建立 V1 编译环境。其 CUDA、驱动及编译器版本沿用服务器已验证环境，需从实际构建日志核实，当前不臆测兼容版本。

### 后续如何更新

可以用现有 SFTP 工具（例如编辑器远程连接、WinSCP 或其他文件传输工具）将本地修改的文件上传覆盖到服务器同一相对路径。不需要每次重新创建 Python 环境；当前没有第三方运行依赖，重新启动分析命令就会读取最新源码。

- 同步源码与配置；新增文件也要同步，源码重命名／删除时明确处理服务器旧文件。
- 保留服务器自己的 `data/`、`outputs/` 和虚拟环境，不用本地目录覆盖或删除它们。
- 刺激仅在新增或修改时上传，DAT、CSV、manifest 始终作为一套同步。
- 在一次分析开始前完成同步；分析运行期间不要覆盖其源码或配置，每次使用新的输出目录。
- 更新模型后从 code/ 执行回归测试；记录使用的源码版本和配置，避免不同实验结果失去对应关系。

若从 WSL 使用 rsync，可参考以下命令，先把 HOST、账户与远端目录替换成实际值（不使用 --delete）：

```bash
rsync -av --exclude='__pycache__/' \
  /mnt/d/A_Gorger/program/MSTD/code/src/ \
  USER@HOST:/home/USER/projects/MSTD/code/src/
```

此例只同步 src/；修改 examples/、configs/、tests/、docs/ 时同样同步对应目录。Windows 原生 PowerShell 未必具备 rsync，直接用 SFTP 上传同样有效。

## 3. Python 环境

如果服务器已有 Conda（或兼容命令），建立独立环境：

```bash
conda create -n mstd python=3.10 -y
conda activate mstd
cd /your/upload/path/MSTD/code
export PYTHONPATH="$PWD/src"
python --version
python -m unittest discover -s tests -v
python examples/analyze_mt_gratings.py --help
```

如果已有 Python 3.10 或以上，也可使用：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
python --version
python -m unittest discover -s tests -v
```

不要直接使用默认 Python 3.8。没有 Conda 或新 Python 时，先由服务器管理员提供独立 Python >=3.10，再执行上述命令。无需 `pip install`；使用 PYTHONPATH 可避免构建依赖联网。可编辑安装属于可选项，需 setuptools>=68。

所有命令在同一环境执行；重新登录后需要重新激活环境和设置 PYTHONPATH。

## 4. 当前刺激与服务器 V1 采集要求

本轮使用 `srcipts/bw_grating_output/sequence_20261019_134626_053/` 的三件套，保留原清单与 CSV 内容：

| 参数 | 值 |
|---|---|
| 图像 | 80×80、单通道黑白 |
| 方向 | 0、45、90、135、180、225、270、315° |
| 方向定义 | x 右、y 下，从右开始顺时针 |
| 速度顺序 | 1.5、0.5、1、2 像素／帧 |
| 每个方向 | 光栅 100 帧，再黑场 50 帧 |
| 总帧数 | 4800 |
| 帧时长 | 50 ms（由 V1 设置，DAT 本身不存帧时长） |
| 完整仿真时长 | 240000 ms |
| 输入 DAT 大小 | 30720025 字节 |

CARLsim 运行配置需要满足：

1. 使用上面的输入 DAT，读取全部 4800 帧，不沿用 80／100／1200 帧旧常量。
2. 图像和 V1 输出群体均为 80×80，输出深度 84，即 28×3。
3. 启用 V1 SpikeMonitor，保存真实 `spk_V1.dat`，而不是仅保存 MT 或网络快照。
4. 每帧推进 50 ms，运行全部 240000 ms，包含黑场与静默尾段。
5. 整份连续 DAT 内不要重置模型历史。每次独立完整实验开始时重置。
6. 记录是否翻转／缩放图像，确认实际链接的修改版 MotionEnergy 库和通道编号布局。
7. 保存输入 SHA256、程序／库／快照版本、时长、帧数、随机种子和运行日志。

当前编号协议为 x 最快、其次 y、其次 28 类滤波器、最后三个尺度。若服务器实际布局不同，需先改适配器，不可只修改方向标签来掩盖错配。

连续清单的 type=0 输入应沿用现有 VisualStimulus 工程实际支持的读帧路径。服务器可执行程序及传参接口未在本机核实，因此这里不提供虚构的 V1 启动命令，继续使用你现有已成功运行的采集入口。

## 5. 检查响应并运行实际 MT 分析

假设响应保存到 `data/raw/grating_run01/spk_V1.dat`。先检查头部：

```bash
python -m mstd inspect-spikes data/raw/grating_run01/spk_V1.dat
```

应看到 width=80、height=80、depth=84、version≈0.2。之后执行：

```bash
python examples/analyze_mt_gratings.py \
  --manifest srcipts/bw_grating_output/sequence_20261019_134626_053/manifest.json \
  --spikes data/raw/grating_run01/spk_V1.dat \
  --projection-config configs/v1_projection.json \
  --projection motionProj1 \
  --warmup-ms 1000 \
  --border 4 \
  --radius 1 \
  --output outputs/grating_mt_run01
```

程序从清单读取总时长和帧时长，完整消耗事件流并保留空窗／静默尾段。它验证输入 DAT 的尺寸和哈希、分段覆盖、V1 网格、事件时间和编号。不要求相机内参，因为这是图像方向测试。

`--warmup-ms 1000` 是每段头 20 帧的统计排除量，不会从 V1 删除这些帧或清零历史。光栅段剩余 80 帧评价，黑场段剩余 30 帧统计；所有帧仍写入 frames.csv 以便检查瞬态。可根据真实启动曲线重新设定，不能默认 1 秒一定充分。

输出目录必须不存在；重复实验换新目录。中途失败可能留下部分 frames.csv，但不会生成成功的 report.json，应先定位错误后使用新目录重跑。

纯标准库实现优先保证可检查性。4800 帧 80×80 MT 分析需逐帧处理，尚无该服务器运行耗时测量；不要据此承诺实时性。程序仅保留当前稠密窗口和小型汇总，逐帧 CSV 直接写盘，无需保存完整84通道稠密历史。

### 长时间没有终端输出时

早期版本仅在全部结束时输出 JSON。新版脚本在启动、第一帧、每100帧和结束阶段向 stderr 打印进度、耗时和粗略剩余时间，并刷新 CSV。可用 `--progress-every 20` 提高频率，0关闭进度；ETA 会随事件密度和服务器负载变化。输入刺激240秒不等于分析墙钟时间240秒。

旧进程正在运行时，先在另一个终端进入同一 code/，观察：

```bash
pgrep -af '[a]nalyze_mt_gratings.py'
top -u "$USER"
watch -n 10 'wc -l outputs/grating_mt_run01/frames.csv; tail -n 1 outputs/grating_mt_run01/frames.csv'
```

CSV最后一行开头的 start_ms/end_ms 表示已写出的仿真时间窗，完整数据应有4800行加表头。旧版 CSV 有缓冲，行数会分批增长；一小段时间不变不等于卡住。高单核 CPU 占用且时间窗持续推进，通常说明还在计算。低CPU且长时间无进展时，再看进程状态和数据盘I/O；没有找到进程时检查原终端报错及是否已经生成 report.json。

若选择停止旧进程，原终端 Ctrl+C 后上传新版 examples/analyze_mt_gratings.py，再使用新输出目录重跑。当前不支持从半截CSV续算；不要在旧进程运行中覆盖脚本并期待它获得新进度功能。单纯增加 `python -u` 不能让旧版出现不存在的中途 print。

## 6. 输出怎样判读

`report.json`：

- `direction_channel_angles_deg`：数组通道实际角度。原配置次序约为 [0,315,270,225,180,135,90,45]，不要按数组索引假定每项递增45°。
- `projected_direction_mean_hz`：MT 之前的八方向投影率均值，检查调谐形状；不是实际 MT 神经元放电率。
- `projected_winner_direction_deg`：投影峰值方向。
- `mt_direction_deg`：实际池化／归一化后 MT 空间平均向量的方向。
- `mt_error_deg`：与刺激方向的最短圆周角误差。
- `mt_evidence`：ROI 内局部 MT 向量长度均值，范围约0～1，不是速度或 Hz。
- `mt_coherence`：平均向量长度／局部向量长度均值，反映空间／时间方向一致性，不是正确概率。
- `speed_results`：每个刺激速度是否具有完整八方向，及方向工程筛查结果。
- `moving_directions_passed`：各速度八方向是否全部通过；**不包含黑场／静态误报验收，不能单独作为完整验收结论**。

`frames.csv` 保留每个50ms窗的投影率、MT方向、证据、一致性、段号和是否进入稳定统计。切换延迟需结合这些时间序列检查，本工具尚未自动拟合延迟。

默认筛查：方向误差≤22.5°，MT证据≥0.02，一致性≥0.1。阈值是初始工程选择，不能替代对静态／黑场、调谐曲线、独立纹理和重复实验的检查。无响应不能因角度偶然接近而通过。

理想结果：输入方向改变时投影调谐峰和 MT 群体方向相应移动；不会总指向同一方向。允许邻近通道共同活动。黑场和速度0没有期望方向，不评价其方向误差，但应单独检查持续响应和残留时间。

当前投影参数来源于旧 C++，方向符号、增益、尺度权重均待实测校准。`motionProj1` 原标称速度1.5px/frame；另两组原标称0.125和9，不直接对应本次0.5、1、2速度，也不代表已实现速度识别。先固定 motionProj1 比较多速度调谐，再做有依据的投影校准；不要为每个测试条件单独挑最优参数后当作泛化精度。

## 7. 如果使用八个独立刺激文件

原 `v1-direction-calibration-v1` 清单可用：

```bash
python -m mstd analyze-calibration /path/calibration_grating/manifest.json \
  --projection-config configs/v1_projection.json \
  --projection motionProj1 \
  --output outputs/independent_projection_run01
```

先在清单每个 run 的 `spikes` 中填写对应响应相对路径，并保留原刺激文件。该命令给出投影方向报告，不能称为实际 MT 完整测试。新连续分析脚本仅接受 v2 连续清单；两种方案不要互换清单。

## 8. 返回哪些结果

保留本次使用的刺激清单、分段 CSV、对应 spk_V1.dat、运行配置／日志、report.json 和 frames.csv。先确认输入匹配与时序，再诊断投影、MT 或前端问题。

本机已通过28项 unittest，包括真实 MT 编码调用、八方向符号、预热排除、分段边界、黑场、反向／静默不通过、完整时长越界和输入校验。测试夹具是人工脉冲，不是实际 V1 响应。Ubuntu 服务器执行与真实神经模型结果仍待现场验证。
