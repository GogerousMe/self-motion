# 方向校准工具交付验证（2026-10-19）

本次按用户选择准备手动Linux采集，不连接服务器。

## 已运行验证

- 19项unittest全部通过，包含此前13项回归。
- 新增实际灰度像素的右／下／左／上平移符号测试，覆盖光栅和纹理。
- 检查VisualStimulus文件头、帧尺寸、重复生成确定性和输入校验。
- 用已知滤波器响应夹具测试8方向报告、校验值错误拒绝、pending不通过。
- 用模拟进程测试采集编排：8个独立响应路径、manifest更新和响应哈希；不输出指定文件时明确失败。
- 两套正式刺激各8段，80×80×80帧，50ms/帧、预热20帧、速度1.5px/frame。
- tar.gz内部22个文件逐个核验：16个刺激DAT、2个manifest、2个PGM预览、采集脚本、说明。
- 每个DAT为512025字节，25字节头后为80×80×80个uint8。
- 无真实响应的分析返回measured_runs=0、pending_runs=8、calibration_passed=false。

## 本地交付物

- outputs/calibration_bundle_01.tar.gz（807099字节）
- outputs/calibration_bundle_01.tar.gz.sha256
- outputs/calibration_grating_01/
- outputs/calibration_texture_01/
- outputs/calibration_pending_report_01/

压缩包SHA256：19bf3a02aefa4cd33915768e26a6ac6107d4a9bdb76e442638ca4ec120f5207c。
刺激包与实验输出位于忽略目录，未纳入Git；源码支持重新生成。

## 当前限制

未在用户Linux环境编译或运行C++ V1。C++接入片段仍需在服务器工作副本按说明整合并编译。
采集脚本的本机测试使用模拟进程，不是神经模型响应。
新刺激没有真实V1输出，因此没有完成方向标定，也未验证自运动精度。
原有grating目录的历史响应尚缺少可靠的视频／配置配对，未强行赋予八方向标签。

下一步：用户按CALIBRATION.md采集，返回两套完整目录及其responses与更新后的manifest；随后执行真实方向校准和增益／符号诊断。
