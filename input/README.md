# 输入数据

先读 [目录与命名规范](../docs/FILE_ORGANIZATION.md)。`registry.csv`登记图像刺激DAT的尺寸、帧数和SHA256，包含reusable输入和fixture验证夹具。

- `stimuli/natural/`：自然视频刺激，一套数据一目录，DAT采用完整刺激ID。
- `stimuli/grating/`、`texture/`、`selfmotion/`：协议套件，外层目录统一命名；内部manifest引用的文件名保留。
- `sources/`：本项目已有原始图像帧。其他原始素材仍位于外部Datasets目录，不重复复制。
- 单纯输入预览跟随刺激包；包含神经响应的视频写入outputs/visualizations。
- 历史sidecar内的路径和文件名不改写；当前位置以索引、artifact.json和迁移映射为准。
