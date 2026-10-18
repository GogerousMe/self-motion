# 修改版 V1 与历史 DAT 文件核对补充

> 2026-10-19 update: SpikeMonitor DAT import and 84-to-8 projection are implemented. See [M1 validation](M1_VALIDATION.md) and [import guide](CARLSIM_IMPORT.md). Earlier notes below describe the v0.1 milestone.

日期：2026-10-18。此文补充并修正 TECHNICAL_PLAN.md 中依据默认源码作出的维度判断。

## 实际源文件

- external/MotionEnergy/cppME/cppME/src/save/motion_energy.cu
- projects/v1mt_wyq/src/v1toCDS1_LoadSimulation.cpp

修改版 calcV1complex() 注释掉 calcV1direction(speed)，直接复制 d_resp_，大小为 W×H×28×3。
因此 84 通道的 V1 群体与这份修改版的输出维度一致；不能继续依据默认源码称其必然错配。
28 个通道是 x/y/t 时空滤波器朝向，不是全圆均匀分布的 28 个平面方向。
内部 nrDirs=8 宏仍用于方向投影，但此输出路径绕过了该投影。
修改版中 speed 不再通过 calcV1direction 选择输出方向投影，不能将该调用参数当成这 84 通道的速度标签。

LoadSimulation 中的连接函数依然出现 proj[i][j]；但运行还会读取网络快照，不能仅凭连接源码断定加载后的实际连接。需要核对快照来源及实际连接。
该文件当前 V1 SpikeMonitor 调用被注释，只有 MT1CDS 监视开启。历史 spk_V1.dat 不一定来自这一份运行，应关联正确视频和配置，防止读取旧结果。

## 已实际读取的文件头（只读取头与首尾事件）

| 相对 v1mt_wyq 的路径 | 群体尺寸 | 最后事件时间 ms | 事件数 |
|---|---|---:|---:|
| results/spk_V1.dat | 80×80×84 | 12599 | 105929046 |
| results/6-2/1/spk_V1.dat | 32×32×84 | 4199 | 82952 |
| results/grating/spk_V1.dat | 32×32×84 | 3999 | 6386984 |
| results/highway/spk_V1.dat | 32×32×84 | 4999 | 3435581 |

文件头与仓库 SpikeReader.m 对应：int32 signature=206661989，float32 version≈0.2，三个 int32 网格尺寸，共20字节。
随后每个事件为两个 int32：[time_ms, neuron_id]，每事件8字节。上述文件均按小端成功解释。
最后放电时间不是仿真总时长，静默尾段必须从配置或原始序列恢复。
这些检查没有遍历全部事件，不能作为完整文件内容验证或前端功能验证。

## 接入修订

```text
CARLsim spk_V1.dat
  → 流式脉冲解析与完整时长／时间窗对齐
  → R[t,scale,filter,y,x]，28×3 通道
  → 带明确投影矩阵的 V1→MT 适配器
  → 8方向 MT 表示
  → 现有 MT–MSTd 参考后端
```

首版 Python 的 v1-rate-v1 是方向率接口，不能用 spk_V1.dat 直接替代 rates.f32。
需要新增 CARLsim 事件解析器与 raw-filter adapter；当前没有宣称它们已实现。
索引应与修改版数组保持一致：x=id%W；y=(id//W)%H；filter=(id//(W*H))%28；scale=id//(W*H*28)。
在群体编号确实遵循此布局时使用，先用可控刺激验证。
50 ms 窗口的事件数 n 对应估计活动率 20n Hz，不能视为精确恢复泊松采样前的响应。
负投影权重须保留抑制作用。直接线性投影加整流是参照适配，不等价复现既有 CARLsim 兴奋／抑制神经元网络。

第一优先使用已有 dat 和对应视频／运行配置，不要求先修改旧工程导出浮点数据。
未来直接导出泊松之前的84通道连续响应作为对照，可区分采样噪声和后端误差。

## 首批刺激

1. 方向校准：8个平移方向，独立序列，先单一可控速度，再扩展速度；验证28→8投影及坐标符号。光栅和随机点互补，光栅含孔径歧义。
2. 自运动基线：静态三维随机点场景，已知相机内参／平移向量，中心和偏左／偏右的膨胀中心，无旋转无动态物体。
3. 动态干扰：固定同一相机轨迹，仅增加独立运动物体与遮挡，对照误差和恢复时间。
4. 真实第一视角：随后加入有相机运动真值的视频；历史跟踪视频可测试读取，不能直接当自运动航向真值数据。

试验宽高可先保持80×80，50ms/帧作为明确的仿真时间映射；原始视频采样时间必须另记。
角度评估需要缩放后相机内参或已知合成相机参数。
