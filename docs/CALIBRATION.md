# Linux 手动采集与方向校准

此阶段校准局部运动方向，不需要相机内参，也不代表自运动航向精度已验证。
Python生成真正的灰度刺激帧，送入你修改的C++ V1；人工滤波器响应只用于单元测试。

## 1. 已准备的刺激

默认每套8个独立序列：80×80、80帧、50ms/帧，共4000ms。速度1.5像素/帧。
前20帧（1000ms）作为预热；分析后60帧。预热长度是初始实验设置，需后续检查是否足够。
角度在图像坐标中顺时针增加：0右、45右下、90下、135左下、180左、225左上、270上、315右上。

- grating：周期12像素的正弦光栅，幅值100、均值127.5。
- texture：固定种子的随机亮点纹理，经双线性重采样作周期平移，用于补充光栅的孔径歧义。
- DAT格式：VisualStimulus v1，25字节文件头，picture类型4、单通道、uint8帧；与已有readFrameChar接口一致。
- 每个序列应启动独立进程，重置V1时间历史；禁止直接拼接为一条连续视频后沿用原清单。

本机生成命令：

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m mstd generate-calibration --output outputs/calibration_grating_01
python -m mstd generate-calibration --output outputs/calibration_texture_01 --kind texture
python examples/package_calibration.py outputs/calibration_grating_01 outputs/calibration_texture_01 --output outputs/calibration_bundle_01.tar.gz
```

输出路径不可重复使用。包内只有刺激、清单、预览、采集脚本及本说明，没有历史V1响应。

## 2. 在Linux解包

把tar.gz及其.sha256一并拷贝到服务器，在同一目录运行：

```bash
sha256sum -c calibration_bundle_01.tar.gz.sha256
mkdir calibration_bundle_01
tar -xzf calibration_bundle_01.tar.gz -C calibration_bundle_01
```

解压目录包含grating/、texture/、collect_calibration.py和COLLECTION.md。
采集脚本使用标准Python库，不要求服务器安装本模型或NumPy。

## 3. 给既有C++入口增加两个可选参数接口

不改V1数学模型。请在服务器独立工作副本修改和编译；源文件示例为v1toCDS1_LoadSimulation.cpp。
以下是接入片段，并非已在本机编译的完整CARLsim程序。

在现有stimPath设置之后、构造VisualStimulus之前加入：

```cpp
// <cstdlib> / <stdlib.h> already provides getenv.
const char* calibrationInput = getenv("V1_CALIBRATION_INPUT");
const char* calibrationOutput = getenv("V1_CALIBRATION_OUTPUT");
if (calibrationInput) stimPath = calibrationInput;
VisualStimulus stim(stimPath); // replace the existing constructor, do not duplicate it
```

在setupNetwork之后，启用V1监视，并替换原有默认MT监视代码：

```cpp
if (calibrationOutput) {
    sim.setSpikeMonitor(gV1, calibrationOutput);
} else {
    // Keep normal-run monitors here as appropriate.
    sim.setSpikeMonitor(gMT1CDS, "DEFAULT");
}
```

核对frameDurMs=50、V1群体为width×height×84、实际链接的是直接输出28×3通道的修改版MotionEnergy库。
从stim.getLength()读取全部80帧；不要使用历史frameReal常量截断。
校准模式不要启用其他默认路径的脉冲输出，以免覆盖历史结果。CARLsim自身日志也可能写默认目录，运行前保存已有results或使用独立工作目录和所需网络快照。
不要直接使用旧run.sh进行采集：其可执行目标名及构建路径不一定对应当前LoadSimulation入口。
先手工确认编译的是哪一个源文件、加载哪个网络快照；这一步不能由目录名代替。

## 4. 批量采集

将下面两处路径替换为服务器实际路径。executable必须是上述修改后编译的入口，workdir必须含它需要的网络快照等相对路径资源。

```bash
cd /absolute/path/calibration_bundle_01
python3 collect_calibration.py --suite grating --executable /absolute/path/v1_calibration_runner --workdir /absolute/path/calibration_workdir
python3 collect_calibration.py --suite texture --executable /absolute/path/v1_calibration_runner --workdir /absolute/path/calibration_workdir
```

如果需要指纹记录，可增加 `--library /absolute/path/libME.a.0.3.0`。
脚本通过环境变量传入每条刺激与唯一响应路径，每条启动独立进程。若程序忽略参数、未输出指定文件或群体不是84通道，会报错而不会回退读取旧文件。
输出位于每套的responses/<run>/spk_V1.dat和process.log；成功后自动更新manifest.json，记录刺激对应关系、响应与可执行文件SHA256。
库指纹只记录你提供的文件，不能代替实际链接审计。脚本不会修改源代码，也不会覆盖已存在的响应目录。
若失败，保留日志和已有响应；修复后使用重新解包的新目录开始完整采集。

若不使用脚本，也可逐条运行并把对应响应路径填入manifest的spikes字段，但务必保留序列标签和运行配置。

## 5. 返回并分析

把完整grating/和texture/目录（含更新的manifest及responses）拷贝回本机，例如outputs/collected_grating_01和outputs/collected_texture_01。

```powershell
python -m mstd analyze-calibration outputs/collected_grating_01/manifest.json --projection-config configs/v1_projection.json --projection motionProj1 --output outputs/grating_report_01
python -m mstd analyze-calibration outputs/collected_texture_01/manifest.json --projection-config configs/v1_projection.json --projection motionProj1 --output outputs/texture_report_01
```

报告包含report.json和directions.csv：每个已知方向的八通道平均响应、84个滤波器平均活动率、胜出方向、环形角误差、方向集中度。
默认排除4像素边界，先逐窗投影再汇总，避免把“平均后整流”错误当成“整流后平均”。
默认工程判据：八方向数据齐全，各方向有响应且集中度>=0.1，胜出方向误差<=22.5°。
这些阈值不代表生物学标准或概率置信度；应联合检查光栅、纹理、重复随机种子和多速度。
没有响应的run为pending，不能通过校准；文件尺寸或刺激校验值错配会报错。
真正的刺激与响应是否来自同一次采集仍依赖采集过程，格式检查不能自行证明这种因果关系。

通过方向校准后，再做带已知相机内参和运动真值的3D随机点自运动测试。
