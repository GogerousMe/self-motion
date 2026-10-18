# Python V1–MT–MSTd 自运动模型

Python 为开发主体；通过离线数值接口对接现有 C++/CUDA MotionEnergy/V1。
当前版本是可运行的纯平移参考原型，不是 Layton 论文的逐式复现，也不是完整自运动或 SLAM 系统。

- [完整技术方案](docs/TECHNICAL_PLAN.md)
- [V1 接口协议与 C++ 接入说明](docs/V1_INTERFACE.md)
- [开发进度与下一步](docs/DEVELOPMENT.md)
- [本机实际验证记录](docs/VALIDATION.md)
- [CARLsim 脉冲 DAT 接入](docs/CARLSIM_IMPORT.md)

## 快速运行

Python 3.10+，首版无第三方运行依赖。NumPy/PyTorch 加速属于后续阶段。
在项目根目录运行（PowerShell）：

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m unittest discover -s tests -v
python -m mstd demo --output outputs/demo_01
python -m mstd run outputs/demo_01/recording --output outputs/replay_01 --last-frame-duration-s 0.05
```

输出路径必须尚不存在，避免覆盖实验。安装环境具备 setuptools 时，也可 `python -m pip install -e .`，随后使用 `mstd` 命令。

演示生成 16×16、8 方向的人工径向响应，经过文件读写、MT、MSTd，输出中心平移方向估计。
这是接口与几何计算的 smoke test；它没有验证真实视频前端、动态物体鲁棒性或曲线运动。

## 当前边界

- 输入：已完成方向投影、全圆均匀方向采样的响应率，单位 Hz。
- 模型：局部池化／归一化、径向膨胀模板、简化分流竞争动力学。
- 输出：投影方位角 alpha_x/alpha_y、活动分布、有效标志和未校准的可靠性指标。
- 当前 CLI 仅扫描水平 -20° 至 +20°，5° 间隔；不输出 yaw、转向状态、曲率或绝对速度。
- v0.2 支持 `inspect-spikes` 与 `convert-spikes`，读取84通道 CARLsim SpikeMonitor DAT，并通过显式28→8投影转成模型输入；用法与限制见上面的接入文档。
- 不支持任意原始浮点滤波器文件、网络快照或 HDF5。
- 没有修改现有 C++ V1 工程；真实导出仍需确认其实际链接库与输出维度。

## 当前开发重点

Python 是模型主体，保留修改版C++ V1。接入真实DAT只证明接口可用；投影符号、增益、视频对应关系和相机内参校准后，才能报告自运动精度。
原始DAT、二进制实验输出与本机数据均不上传；测试使用临时生成的小型夹具。
