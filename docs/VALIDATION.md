# 首版验证记录

> 2026-10-19 update: SpikeMonitor DAT import and 84-to-8 projection are implemented. See [M1 validation](M1_VALIDATION.md) and [import guide](CARLSIM_IMPORT.md). Earlier notes below describe the v0.1 milestone.

日期：2026-10-18，Asia/Shanghai。
实际项目位置：D:/A_Gorger/program/MSTD/code。

## 已执行

1. `python -m unittest discover -s tests -v`：7 项全部通过。
   覆盖协议回读与通道顺序、拒绝负值／非有限值／错误维度／重复时间戳、拒绝截断文件及错误编码、左右平移方向、静默和各向同性输入无效、动态活动有界、积分步长收敛及重置。
2. `python -m mstd demo --output outputs/demo_01`：20 帧，最终 valid=true、alpha_x_deg=0.0。
3. `python -m mstd run outputs/demo_01/recording --output outputs/replay_01 --last-frame-duration-s 0.05`：与演示的 estimates.json 逐项相等。
4. `g++ -std=c++11 -Wall -Wextra -pedantic examples/export_fixture.cpp -o outputs/export_fixture.exe`：独立 C++ 示例编译成功。
5. 执行 C++ 示例后，Python 读取 2 帧、每帧 16 个 float32，与期望数值 0..31 逐项相等。

环境为本机现有 Python 与 MSYS2 UCRT64 g++。使用 PYTHONPATH=src 运行，未验证 pip 安装流程。

## 结果位置

- outputs/demo_01/recording/：人工 V1 方向响应输入。
- outputs/demo_01/estimates.json：逐帧结果、活动与模板分数。
- outputs/demo_01/run_config.json：本次配置与输入元数据。
- outputs/replay_01/：离线回放结果。
- outputs/cpp_fixture/：独立 C++ 协议样本。

outputs/ 已加入忽略规则，实验目录不覆盖。重新运行请选择新的输出路径。

## 结论边界

已确认：参考代码能运行、文件协议可以跨 C++/Python、简单合成输入上的几何符号与数值行为符合预期。
未确认：真实 V1 输出正确性、真实相机标定、原 CD 论文复现、动态物体鲁棒性、旋转／曲率解码、实时性。
合成响应与径向模板具有相近假设，因此 demo 不构成泛化性能证据。
