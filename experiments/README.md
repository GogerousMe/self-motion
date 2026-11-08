# 实验管理

遵守 [目录与命名规范](../docs/FILE_ORGANIZATION.md)，新实验创建唯一目录并登记 `registry.csv`。

- `scratch/`：试跑、调参、排错和临时渲染。
- `validation/`：校准、受控检查、消融和回归；保持自包含验证包完整。
- 正式归档响应保存到outputs/recordings，单运行展示到outputs/visualizations，多运行对比到outputs/comparisons。
- legacy_imported只表示已迁移，不代表已通过或可作论文结论。
