# CARLsim MT 与 Python Stage 2 对比视频

脚本 `examples/compare_carlsim_python_mt.py` 读取真实 MT SpikeMonitor DAT（深度8）、统一模型的 `mt_pooling` 逐帧监视器和原图，输出三行 MP4：原始彩色帧、CARLsim MT 实际脉冲率、Python Stage 2 池化方向响应率。Python Stage 2 不是实际 MT 放电率。需要 NumPy、Pillow、PATH 中的 ffmpeg；无需重新运行模型或安装 PyTorch。

在项目根目录运行（PowerShell）：

```powershell
$env:PYTHONPATH = 'src'
python examples/compare_carlsim_python_mt.py --spikes outputs/recordings/daria_walk/legacy__daria_walk__carlsim_import__r01/spikes/spk_MT1CDS.dat --recording outputs/recordings/daria_walk/legacy__daria_walk_180x144_f000084__py_rate_cuda__r01 --images D:/A_Gorger/program/Datasets/daria_walk --stimulus input/stimuli/natural/daria_walk/daria_walk__180x144__f000084__dt50ms__gray_native__v01/daria_walk__180x144__f000084__dt50ms__gray_native__v01.dat --output experiments/scratch/manual__daria_walk_mt_comparison_new__r01
```

输出目录必须尚不存在。`--stimulus` 为 Python 运行的确切 DAT，脚本验证其 SHA256，并逐帧验证原图转灰度像素与 DAT 一致，不会自动缩放或裁剪原图来伪造配对。

时间窗为 `[i*50,(i+1)*50)` ms，率为计数/窗口秒数；可用 `--frame-ms` 指定 CARLsim 帧时长假设，必须与 Python 元数据一致。默认零时刻对齐，不自动寻找最佳时移，不添加时间平滑。

CARLsim 文件未包含刺激身份和总仿真时长。已知完整时长后可指定 `--carlsim-duration-ms 4200` 等真实数值，纳入末尾静默；不能根据 Python 时长猜测这个参数。默认只使用最后脉冲所在窗之前的事件支持区间，并在报告标为推断值。视频取两者共同长度，不把缺失记录补零。

CARLsim 原通道角度默认 `[0,315,270,225,180,135,90,45]`（图像坐标，顺时针从右），依据本地源码注释；可用 `--carlsim-directions` 显式替换八个角度。Python 根据监视器方向元数据排序。显示统一为0、45、90、135、180、225、270、315度；不按结果自动挑选映射。

每行使用各自整段99.5百分位固定色标，可用 `--rate-max 640 222` 显式指定两行上限，或指定相同上限观察幅度。超出色标的数值仅显示截顶，CSV和统计保持原值。空间图使用最近邻显示并保留宽高比例；不对两种分辨率进行逐像素比较。

方向均值默认排除各自边界4个原生像素（`--border`），这不代表两个网格上的相同物理感受野。汇总默认排除前1000ms（`--warmup-ms`）；视频和CSV仍保留全部共同帧。逐帧CSV保存八方向空间平均率、最强方向及归一化方向分布总变差距离；静默不强行指定方向。JSON保存输入哈希、时间和方向假设、网格及汇总结果。没有运动真值，不输出方向正确率。

本次数据：CARLsim为80×80，Python为180×144、84帧、50ms/帧。CARLsim最后脉冲3599ms，默认生成72帧、20FPS、3.6秒视频。原图84帧与Python DAT逐像素匹配，但CARLsim输入预处理、方向校准和真实仿真时长未独立证实。不同分辨率改变像素速度和滤波感受野，结果仅用于探索，不足以归因于MT算法。

交付文件：`mt_comparison.mp4`、首帧/预热末附近帧/末帧PNG、`direction_profiles.png`、`direction_statistics.csv`、`comparison_report.json`、`ffmpeg.log`。统计中的胜出方向一致率是两模型之间的一致性，不是正确率；方向总变差距离在0到1之间，0表示分布一致。
