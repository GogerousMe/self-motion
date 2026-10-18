# 开发记录

> 2026-10-19 update: SpikeMonitor DAT import and 84-to-8 projection are implemented. See [M1 validation](M1_VALIDATION.md) and [import guide](CARLSIM_IMPORT.md). Earlier notes below describe the v0.1 milestone.

## 2026-10-18：M0 首版

已实现：
- Python 包结构和 CLI，标准库参考后端。
- V1-rate-v1 数据协议、严格读取和写入。
- MT 局部池化、归一化和空间方向群体向量。
- 径向膨胀 MSTd 模板、简化循环竞争、有效性门控。
- 合成方向响应、完整文件回读 demo、逐帧活动和配置输出。
- C++ float32 小端写出头文件示例（尚未集成旧工程）。
- 接口、几何方向、无运动、积分步长收敛等测试。

未实现：真实 V1 数据接入、速度通道、论文原始 CD 方程、真实视频、曲率／旋转解码、动态分割、ESN、脉冲回放、GPU 加速。

下一步 M1：
1. 在实际运行 V1 的环境核对 libME 安装版本与源码，确认输出 8 方向还是 84 滤波器通道。
2. 用向右、向上、向左、向下的已知运动刺激采集响应，记录帧率和滤波器启动延迟。
3. 根据结果选择方向接口或新增 raw-filter adapter，导出一个短序列。
4. 对真实输入做二进制回读、逐通道图和几何对齐验证，再开展性能结论。

本次未修改原有 C++ 工程。当前方向判据和数值测试不能代替真实前端与动态场景验证。

实际执行结果见 VALIDATION.md：7 项测试通过、20 帧 demo 与重放一致、独立 C++ exporter 编译及跨语言数值回读通过。
