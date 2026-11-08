# 项目会话入口

## 开始任务前

1. 阅读 `docs/FILE_ORGANIZATION.md`，这是目录、命名与产物登记的统一规范。
2. 阅读与任务相关的运行说明和现有源码；查看 `git status --short`，保留用户已有修改。
3. 创建数据前先查 `input/registry.csv`，创建实验前先查 `experiments/registry.csv`。
4. 历史路径通过 `docs/migrations/20261107_layout/paths.csv` 查询；历史报告内的绝对路径不是当前路径依据。

## 执行要求

- 先明确输入刺激ID、运行ID、模型后端、参数、保存位置与父实验，再运行。
- 可复用刺激写入 `input/stimuli/`；试跑写入 `experiments/scratch/`；专项验证写入 `experiments/validation/`。
- 正式响应写入 `outputs/recordings/`；单运行可视化写入 `outputs/visualizations/`；跨运行比较写入 `outputs/comparisons/`。
- 编译写入 `build/`，临时状态写入 `cache/`，发布包写入 `exports/`；禁止把这些内容放在outputs根目录或源码目录。
- 使用规范编号和全新目录，禁止使用final/latest/new作为版本管理。不要覆盖旧输出。
- 保留原生录制格式、完整metadata、帧号和时间戳；没有证据不补写帧率、真值、通过状态或模型等价性。
- 新增或变更产物须更新对应registry，并添加 `artifact.json` 关联输入、运行和展示配置。格式见规范。
- 完成后核对产物路径、输入哈希、原生加载和必要测试。不要将大数据、本机状态或环境提交Git。
- 禁止仅按.dat扩展名归类：图像刺激、V1/MT脉冲与中间数据属于不同类型。
- 历史验证套件内部的刺激是允许保留的自包含夹具；不要直接拆散或改名破坏引用。

未涉及本轮任务的既有代码修改不得回滚。没有用户指示不要提交或推送Git。
