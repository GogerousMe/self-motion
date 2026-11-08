# 20261107文件整理与验证记录

统一规则见 [文件存放与命名规范](../../FILE_ORGANIZATION.md)。工作区根目录和code仓库均已添加AGENTS.md作为新会话入口。

## 迁移结果

- 原outputs根目录仅保留recordings、visualizations、comparisons三类子目录。
- 自然视频刺激采用数据集、宽高、帧数、帧间隔、处理方式和版本的完整名称；多条件套件保留协议内文件名，外层目录统一命名。
- 刺激、模型响应、单运行展示、跨运行比较、验证、试跑、构建、本机状态和上传包已分开存放。
- 287项初始移动操作，1862个原文件，共10719701574字节，迁移前后全部核对SHA256。原始报告和二进制内容不变。
- `input/registry.csv`登记102个图像DAT（包括自包含验证夹具）；`experiments/registry.csv`登记37个历史产物和3个本次运行验收产物。
- v03连续光栅包原DAT名与manifest不一致；在确认内容SHA256与manifest完全一致后，将协议文件名恢复为all_stimuli.dat，操作另记journal，paths索引已更新。
- 新的CMake目录build/carlsim_mt已重新配置和编译；运行库复制至程序目录。原编译树保持为legacy快照，不作增量构建目录。

## 验证证据

- 路径迁移、输入身份保护、连续率可视化、脉冲可视化和模型运行相关13项unittest通过。
- 使用迁移后的真实Daria响应重放2帧视频，无需重算模型；ffprobe确认2帧。
- 从分离后的连续光栅stages.json/stages.f32重放2帧视频，确认响应与展示分离后仍可使用；ffprobe确认2帧。
- 新构建的真实CARLsim程序7项受控运行检查通过。
- 源码打包检查确认包含AGENTS、规范、路径兼容模块及原生CMake文件，排除input、outputs、experiments、cache、exports及迁移审计目录。
- Git diff空白检查通过。MATLAB命名入口已检查和更新，本轮未启动MATLAB执行；服务器数据未移动。

## 保留的待核对事项

1. Daria corrected版对比报告记录的CARLsim脉冲SHA256，未匹配到当前本机导入文件。
2. Snowboard对比报告记录的CARLsim脉冲SHA256，未匹配到当前本机导入文件。
3. 原outputs根目录散落的mt_stages.mp4尚未可靠确认输入与父响应，保存在visualizations/unresolved。

这些产物已保留，artifact中登记缺失关联；不能据目录名补造输入身份或验收结论。外层tmp、参考文献及PDF保持原位，新实验按新规则创建。

## 审计文件

- `paths.csv`：逐文件旧位置、当前位置、大小、SHA256和类别；查找旧名字优先使用此表。
- `plan.json`：初始批量迁移计划。
- `journal.jsonl`：实际完成操作及后续协议文件名修复。
- `verification.json`：完整原文件SHA256验证结果。
- `associations.json`：首次追加关联信息的统计。
- `final_audit.json`：最终索引与路径完整性检查。
- `runtime_checks.json`：真实重放视频与原生引擎运行检查结果。

迁移工具为本次一次性脚本，不应在已有数据上重复执行。回溯操作前应查journal和paths，禁止用旧脚本覆盖当前目录。
