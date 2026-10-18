# C++ V1 → Python 数据协议 v1

> 2026-10-19 update: SpikeMonitor DAT import and 84-to-8 projection are implemented. See [M1 validation](M1_VALIDATION.md) and [import guide](CARLSIM_IMPORT.md). Earlier notes below describe the v0.1 milestone.

> 修改版 V1 与历史 spk_V1.dat 已核对，详见 [源码与DAT补充](V1_SOURCE_CORRECTION.md)。以下协议仍是方向率接口，不能直接读取该脉冲DAT。

## 当前支持

每段独立序列一个目录：metadata.json + rates.f32。
方向响应率接口，不是脉冲文件，不是原始 28×3 滤波器。

```json
{
  "schema": "v1-rate-v1",
  "encoding": "direction_rates",
  "dtype": "float32-le",
  "layout": "TDYX",
  "units": "Hz",
  "coordinates": "x-right_y-down",
  "timestamp_reference": "frame_start",
  "sample_hold": "until_next_timestamp",
  "width": 2,
  "height": 2,
  "directions_rad": [0.0, 1.5707963267948966, 3.141592653589793, 4.71238898038469],
  "intrinsics_px": {"fx_px": 2.0, "fy_px": 2.0, "cx_px": 0.5, "cy_px": 0.5},
  "timestamps_s": [0.0, 0.05],
  "provenance": {"source": "cpp_v1", "library_version": "TO_BE_VERIFIED"}
}
```

rates.f32 应包含 2×4×2×2=32 个 float32，共 128 字节。
零基索引：offset=(((t*D+d)*H+y)*W+x)，字节偏移=4*offset。
方向单位 rad，在图像坐标系中 0=右，pi/2=下，pi=左，3pi/2=上。
通道顺序可以不同，但 directions_rad 必须逐通道对应；参考解码器要求全圆等间距采样。

已有 C++ 注释写的是 right, up-right, up ...，如果实际输出如此，则其角度应映射为 [0,-pi/4,-pi/2,...]，不能直接照搬 demo 的顺时针顺序。
必须用单方向刺激验证注释与实际行为一致。

## C++ 导出位置

在已经确认语义的 calcV1complex() 输出处、泊松采样之前导出。
若结果在 GPU 上，先显式复制到大小经过验证的 CPU float 缓冲区，再调用 examples/v1_export.hpp 的函数；它不接收设备指针。
导出每帧实际响应 count，不能用声明了 84 通道的群体大小替代真实输出大小。
头文件只负责二进制字节写入；metadata.json 由调用方依据真实相机、通道、时间生成，示例元数据不可当真实标定。
先解决旧工程实际链接库的 8 vs 84 通道问题，禁止通过补零掩盖错配。

## 校验和时间

响应必须非负且有限，二进制大小必须准确，时间戳严格递增。
图像缩放／裁剪后内参同步变换；物理方位角依赖有效标定。
当前输入表示帧起点与保持区间，最后一帧区间在运行命令中显式给出。
真实前端的累计延迟、预热、源时间定义写入 provenance，评估时做因果对齐。
独立序列重置所有状态，同一序列保持历史。

## 后续协议

原始滤波器需要 channel_table（滤波器、尺度、方向／速度标定），使用独立 schema 和 adapter。
脉冲需要 neurId→通道／位置映射、精确放电时间、群体大小及采样定义。恢复活动率时采用因果滤波并计入延迟。
多速度接口将新增速度轴及明确单位，不能把当前 D 轴塞入多个速度并继续沿用本协议。
