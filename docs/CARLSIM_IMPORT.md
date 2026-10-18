# CARLsim DAT 接入（v0.2）

已支持 SpikeMonitor v0.2，大小端识别、分块事件读取、编号／时间合法性检查、静默窗口保留，以及84通道滤波器到8方向的显式投影。
不读取 input 刺激 DAT 或 loadSimulation 的网络快照。扩展名相同不代表格式相同。

## 运行

在项目根目录设置 `PYTHONPATH=src`。PowerShell 示例：

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m mstd inspect-spikes "YOUR_PATH/spk_V1.dat" --max-events 10000
python -m mstd convert-spikes "YOUR_PATH/spk_V1.dat" --output outputs/converted_01 --projection-config configs/v1_projection.json --projection motionProj1 --duration-ms 4200 --bin-ms 50 --intrinsics YOUR_INTRINSICS.json
python -m mstd run outputs/converted_01 --output outputs/inference_01 --last-frame-duration-s 0.05
```

4200ms 仅为命令示例，必须替换为真实仿真总时长；不能依据最后放电推定。
intrinsics.example.json 为32×32人工内参，只用于运行检查，绝不是已有视频标定。请复制并填入输入缩放后的真实内参。
若 duration 不是 bin 的整数倍，末窗使用实际时长。例如 duration=101ms、bin=50ms，末窗1ms；run 的 last-frame-duration-s 应为0.001，程序检查与转换元数据一致。

## 事件与编号

头部20字节：int32 signature=206661989，float32 version=0.2，int32 W,H,depth。
事件为 int32 time_ms, neuron_id。时间非负且不下降，允许同一毫秒多个事件。
适配器只接受 depth=84，输入时间范围严格为 [0,duration_ms)，越界报错，避免静默截断。
当前解析以固定块读取，时间分箱仅保留一个窗口的计数，转换逐帧写盘；不会一次性加载全部事件或全部84通道响应。
模型 run 同样逐帧读取方向率文件，但会保留小规模的逐帧推断结果及时间戳供最终JSON输出。

```text
x = neuron_id % W
y = (neuron_id // W) % H
filter = (neuron_id // (W*H)) % 28
scale = neuron_id // (W*H*28)
```

窗口 [start,end) 的计数除以真实窗长得到Hz。数据只有窗口结束后才可获得。
离线用此平均率在窗口内积分，是对原始脉冲驱动的近似；输出标记窗口结束，不能宣称是窗口开始时已可用的在线估计。

## 投影

配置 configs/v1_projection.json 从用户指定 v1toCDS1_LoadSimulation.cpp 的 motionProj1/2/3 原值提取，包含源文件SHA256。
三组原始标签速度分别为1.5、0.125、9 px/frame，它们是原权重标签，不是对当前数据真实速度的检测结果。
方向角按原代码注释映射为 [0,-pi/4,-pi/2,...]，采用图像 x右、y下坐标，仍需已知运动刺激验证符号。
默认三个尺度等权相加、gain=1，是明确的参考配置；没有拟合原突触权重或神经元增益。

```text
drive[d,y,x] = sum_scale,filter(scale_weight * W[filter,d] * rate[scale,filter,y,x])
output[d,y,x] = max(0, gain * drive[d,y,x])
```

负权重参与完整求和后才整流。不要先删除负数，也不要将28滤波器标签当均匀方向。
该线性投影不是原CARLsim兴奋／抑制细胞、突触延迟和膜电位动力学的复现。
转换输出元数据保存实际投影参数、配置校验值、窗长、完整时长和限制。

## 验证层级

已有真实文件可验证格式、事件合法性与数据流。必须关联原视频、仿真配置和标定后才评估航向精度。
建议采集8方向可控光栅／随机点刺激校准投影，再采集已知纯平移三维点云视频测试MSTd，随后增加动态物体。
原始DAT和实验输出不提交仓库。测试通过程序生成小型夹具，无需下载私人数据。
