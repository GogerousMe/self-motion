# MSTD项目文件存放与命名规范

生效日期：2026-11-07。适用于本机、服务器及所有后续会话。仓库根目录为 `MSTD/code/`；下文路径均相对此目录。

本规范将输入、实验过程、正式响应和展示分开管理，同时通过稳定ID、相对路径与SHA256保持关联。会话先读取 `AGENTS.md`，再读取本文。历史迁移清单位于 `docs/migrations/20261107_layout/`。

## 1 目录职责

| 目录 | 内容与约束 |
| --- | --- |
| `input/sources/<dataset>/` | 原始视频、图像帧或外部数据位置说明；已在Datasets保存的大数据不重复复制 |
| `input/stimuli/<category>/<dataset或bundle>/` | 模型可直接读取的刺激DAT及帧序、分段、内参、真值、生成参数；category为natural/grating/texture/selfmotion |
| 刺激包内的 `preview/` | 只播放输入画面的MP4、GIF、选帧；不包含神经响应 |
| `experiments/scratch/<run_id>/` | 排错、调参、少量帧试跑、渲染夹具、未完成尝试；临时不等于可以擅自删除 |
| `experiments/validation/<run_id>/` | 方向校准、零输入、消融、回归、后端一致性等专项验证；保留完整证据 |
| `outputs/recordings/<dataset>/<run_id>/` | 需要长期保存的V1、MT、MSTd响应及原生报告、配置、监视器元数据 |
| `outputs/visualizations/<dataset>/<run_id>/<view_id>/` | 单次运行的响应图、曲线、刺激与响应同步视频、绘图统计 |
| `outputs/comparisons/<dataset>/<comparison_id>/` | 多个运行或模型的对照图、视频、指标与比较报告 |
| `outputs/visualizations/unresolved/` | 来源尚不能可靠确认的历史展示文件，必须说明未知项 |
| `build/` | 编译树、可执行程序、构建日志；历史CMake树位于legacy快照，含旧绝对路径，不作为可直接增量构建的目录 |
| `cache/` | 可重建缓存与本机软件状态；MATLAB偏好、证书不上传或提交 |
| `exports/` | 发送服务器或分享的压缩包及校验值；目录内原数据仍是权威来源 |
| `configs/` | 可复用参数模板；实际实验参数快照与运行结果一起保存 |
| `src/`、`examples/`、`matlab/`、`srcipts/`、`native/` | 源码与入口脚本；不放新生成数据；srcipts是历史目录名，本次不改源码路径 |
| `tests/` | 自动化测试代码；临时夹具使用临时目录，需要保留的验证写入experiments |
| `docs/` | 使用说明、研究计划、阶段结论、规则与迁移审计，不存大型响应数组 |

外层 `MSTD/参考文献/`、PDF和 `MSTD/tmp/` 中的历史资料不在本次迁移范围；新运行不得再写入外层tmp。

## 2 通用命名

- 新目录与数据文件使用小写ASCII、数字、下划线；用双下划线 `__` 分隔字段，字段内部用单下划线。
- 分辨率统一写成宽×高，例如 `854x480`，不能颠倒；帧数为 `f000070`，至少六位。
- 帧间隔统一写 `dt50ms`；小数点写p，例如 `dt33p333ms`、速度 `s1p5`。必须记录实际采用的数值，不能根据DAT头或文件名猜测帧率。
- 数据版本为 `v01`，重复运行编号为 `r01`。日期时间使用北京时间 `YYYYMMDD_HHMMSS`。无法确认历史运行时间的使用 `legacy`，不把迁移日当作实验日。
- 禁止依赖final、latest、new、最终版等名称识别权威版本；状态在registry中记录。
- 自动刺激生成器可在版本后追加 `__tYYYYMMDD_HHMMSS_SSS` 防止重名；MATLAB点击Run入口已采用此方式。
- 运行手册里的 `manual__...__r01` 是短命令模板；首次试跑可使用，后续运行改用真实北京时间/新编号。它不是可反复覆盖的目录。
- 修改像素、截帧范围、帧序、帧间隔、缩放/补边、真值或参数都要创建新版本/新运行，不覆盖旧文件。

## 3 刺激ID与文件名

单段、可独立使用的刺激ID：

```text
<dataset_sequence>__<width>x<height>__f<frames>__dt<interval>ms__<pixel_geometry>__v<version>
```

示例：

```text
daria_walk__180x144__f000084__dt50ms__gray_native__v01
daria_walk__188x144__f000084__dt50ms__gray_pad__v01
snowboard__854x480__f000065__dt50ms__gray_native__v01
```

独立刺激使用相同的目录名与DAT文件主名，配套元数据为 `<stimulus_id>.dat.json`：

```text
input/stimuli/natural/snowboard/<stimulus_id>/
  <stimulus_id>.dat
  <stimulus_id>.dat.json
  artifact.json
  preview/
```

`dataset_sequence`要能区分原始序列；帧区间、裁剪区域、像素格式、输入图像哈希、坐标约定和完整参数放在元数据中。帧数从DAT头和字节数核实，不能只读旧文件名。原始帧命名及manifest不随意改动，防止帧序失配。

### 多条件套件兼容规则

光栅、自运动等生成器输出的是一个协议包。套件目录使用统一字段，内部由manifest引用的 `grating_d000.dat`、`cloud_forward.dat`、`all_stimuli.dat` 等固定文件名保留。目录内 `f000080` 表示每条件80帧；合并序列目录的 `f004800` 表示整个DAT4800帧；混合尺寸或长度使用 `mixed` 并以manifest为准，不填一个虚假的共同数值。

```text
input/stimuli/grating/grating_sequence__80x80__f004800__dt50ms__v02/
  all_stimuli.dat
  all_stimuli_segments.csv
  manifest.json
  artifact.json
```

套件内条件文件允许短名，这是协议兼容规则；不单独改名或拆走DAT、CSV、manifest、内参、真值。新增生成器也须遵守“独立刺激完整命名、套件统一外层命名”的约定。一次性验证自行生成的夹具可留在验证目录，但必须被input索引标为fixture，不能自动当作正式数据。

## 4 运行与输出命名

运行ID格式：

```text
<time或legacy>__<sequence>_<width>x<height>_f<processed_frames>__<model_backend>__r<repeat>
```

例如 `20261107_143000__snowboard_854x480_f000065__py_rate_cuda__r01`。
模型后端使用 `py_rate_cpu`、`py_rate_cuda`、`py_spike_cpu`、`py_spike_cuda`、`carlsim_cpu`、`carlsim_gpu`；这里的cpu/cuda描述实际设备，不代表算法等价。历史导入信息不足时使用 `carlsim_import`，不猜测设备或处理帧数。

运行目录保存完整原生输出，不给每个 `values.f32` 重复添加长前缀：

```text
outputs/recordings/<dataset>/<run_id>/
  artifact.json
  model_config.json
  model_report.json
  frames.csv
  monitors/<layer>/values.f32
  monitors/<layer>/metadata.json
```

以上为统一模型示例。CARLsim可保留 `spikes/spk_MT1CDS.dat`，其他后端继续使用其原生协议；不要将响应DAT当作图像刺激。Hz、无量纲活动、膜电位、实际脉冲事件分别记录，不能重命名造成物理含义混淆。

默认先在scratch/validation试验，确认需要长期保存后再将完整运行归档。归档不等于passed；未完成和失败实验也可保留，并记录状态。多个目录不能因内容相似就擅自合并或去重。

## 5 可视化与比较

单运行展示ID：`<purpose>__vNN`，例如 `mt_stages__v01`、`mt_spikes__v01`。改变色标、选帧、显示阶段、布局或绘图参数时增加展示版本，不复制或覆盖原始响应。

```text
outputs/visualizations/<dataset>/<run_id>/<view_id>/
  artifact.json
  visualization.json 或 video_config.json
  mt_stages.mp4 或 mt_spiking.mp4
  frames/ 或原生脚本生成的帧图
  direction_statistics.csv
```

`artifact.json`登记运行ID、刺激ID、相对路径；原生绘图配置登记帧范围、采样间隔、色标、方向顺序、时间对齐和单位。当前脚本生成的标准文件名保留。

多运行比较ID为 `<time或legacy>__<sequence>__<comparison_method>__vNN`。比较目录必须列出全部父运行，说明输入是否一致、尺寸/帧率差异及是否允许像素级比较。不同网格不能仅凭视频外观声称数值等价。

若绘图目录中的数组是唯一保留的响应证据，将数组和配套metadata归入recordings；只有可由已归档响应重建的临时数组才属于cache。是否可重建不明确时保留，不删除。

## 6 索引与关联

`input/registry.csv`登记每个刺激的相对路径、宽高、帧数、通道数、SHA256和scope（reusable或fixture）。它不是帧率来源；帧率以配套元数据或实验配置为准。

`experiments/registry.csv`每个逻辑产物一行，登记ID、类型、相对路径、旧路径、状态、父运行ID、刺激ID和备注。状态可为planned/running/completed/failed/partial/legacy_imported/unresolved，只有证据支持时才写passed结论。

新增产物须有 `artifact.json`，最少包含：

```json
{
  "schema": "mstd-artifact-v1",
  "id": "20261107_143000__snowboard_854x480_f000065__py_rate_cuda__r01",
  "kind": "recording",
  "status": "completed",
  "stimulus_ids": ["snowboard__854x480__f000065__dt50ms__gray_native__v01"],
  "input_paths": ["input/stimuli/natural/snowboard/<stimulus_id>/<stimulus_id>.dat"],
  "input_sha256": "实际计算的SHA256",
  "parent_run_ids": [],
  "notes": "参数、设备、随机种子及帧数见原生config/report"
}
```

示例中的占位值不可照抄到正式记录。需要复现时保存原生配置、源码版本/必要源码哈希、设备、随机种子及运行命令。链接源目录使用项目相对路径；外部数据路径单独标记。共享或上传包中不得出现本机证书、虚拟环境和无关数据。

## 7 历史迁移及路径兼容

2026-11-07迁移：独立自然刺激改为完整名称；套件只改外层目录；原始报告、manifest、CSV、二进制与历史压缩包保持字节不变。

`docs/migrations/20261107_layout/paths.csv`逐文件记录旧路径、新路径、大小和SHA256；`plan.json`及 `journal.jsonl`记录操作；`verification.json`记录完整性校验。原始元数据中的旧文件名/绝对路径属于历史证据，不直接批量改写。当前位置由索引和新增artifact说明。

模型可视化入口通过 `mstd.project_paths.resolve_stimulus`读取迁移索引并按SHA256查找刺激，最终仍执行原有输入校验。也可显式传入 `--stimulus <当前DAT路径>`。它不会恢复缺失的服务器文件，也不代表服务器已迁移。

未明确配对的历史视频放unresolved；包含响应缓存的视频目录已按内容分离；历史CMake快照不能直接增量构建。新构建使用 `build/carlsim_cpu/`、`build/carlsim_mt/` 等全新目录并重新配置。

## 8 每次任务的完成检查

1. 检查是否已有同一刺激/实验，确定唯一ID和未占用目录。
2. 明确刺激帧序、帧间隔、几何处理、输入哈希及父实验。
3. 写入对应目录，不在outputs根目录、源码目录和外层tmp散放文件。
4. 保存原生元数据并补充artifact，更新两个registry中相关行。
5. 核对输入哈希、文件大小、运行完成状态；绘图必须能追溯到原响应。
6. 迁移先检查绝对路径位于工作区、目标不存在，保留旧新映射并校验。未经明确要求不删除旧实验、不覆写数据、不提交大文件。

生成数据、缓存、实验和发布包默认不入Git；提交源码、规则、必要索引和轻量审计文件。新增目录时同时检查 `.gitignore` 与打包工具，防止素材、脉冲事件、证书或迁移后的实验被意外上传。
