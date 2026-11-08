# 三段自然刺激的Python脉冲MT运行约定（2026-11-08）

入口：`tools/run_natural_spiking.py`。先按input/registry.csv查找独立刺激、验证DAT头/大小/SHA256及配套.dat.json的frame_ms；不猜帧率、不改像素、不自动缩放。
运行前打印运行ID、实际处理帧数、配置和输出位置；逐帧显示进度。`--dry-run`只做预检，不运行、不创建产物、不登记实验。

## 本轮默认输入

| --sequence | 刺激ID | 宽×高 | 全部帧数 |
|---|---|---|---:|
| daria_walk | daria_walk__180x144__f000084__dt50ms__gray_native__v01 | 180×144 | 84 |
| paragliding | davis_paragliding__854x480__f000070__dt50ms__gray_native__v01 | 854×480 | 70 |
| snowboard | snowboard__854x480__f000065__dt50ms__gray_native__v01 | 854×480 | 65 |

索引中另有Daria补边188×144和snowboard 66帧版。只有显式`--stimulus-id`指定这些已登记ID时才切换，不能把结果混用同一输入ID。
工具默认处理完整输入；--max-frames用于明确的前缀试跑，运行ID写实际处理数量。

## 服务器准备

使用现有Python环境，从code/根目录执行：

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -c "import numpy, matplotlib; print('dependencies OK')"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
ffmpeg -version
```

同步最新src/mstd/、configs/model_python_spiking.json和v1_projection.json、tools/run_natural_spiking.py、三个绘图脚本examples/visualize_python_spiking.py、render_model_mt.py、visualize_mt_gratings.py、input/registry.csv，以及表中三个完整刺激包（DAT/配套JSON/artifact）。
服务器experiments/registry.csv应保留自身实验记录，不能用本机版本覆盖。缺少时工具在实际运行登记时自动建立，不在dry-run时创建。
不需要CMake、CARLsim库、本机build/cache/outputs或测试数据。

原文件仍在服务器旧位置时，先按规范同步对应刺激包到索引路径；本工具不会移动或删除旧文件，也不会把缺失的DAT伪造为已迁移。

## 建议先前缀试跑

```bash
python tools/run_natural_spiking.py --sequence daria_walk --stage scratch --max-frames 10 --repeat 1
python tools/run_natural_spiking.py --sequence paragliding --stage scratch --max-frames 10 --repeat 1
python tools/run_natural_spiking.py --sequence snowboard --stage scratch --max-frames 10 --repeat 1
```

试跑响应位于experiments/scratch/<run_id>/。可视化位于outputs/visualizations/<dataset>/<run_id>/mt_spikes__v01/。
两段854×480输入较大，CPU仿真与脉冲保存可能很慢。仅连续V1 float32记录就约为paragliding 8.98GiB、snowboard 8.34GiB，尚未计入脉冲、其他神经状态与视频。不要把32×32耗时直接套用到这些原生分辨率。

## 正式运行与视频

```bash
python tools/run_natural_spiking.py --sequence daria_walk --stage recording --repeat 1
python tools/run_natural_spiking.py --sequence paragliding --stage recording --repeat 1
python tools/run_natural_spiking.py --sequence snowboard --stage recording --repeat 1
```

每条命令依次运行原生模型、校验完成帧数/输入哈希、调用原生可视化脚本生成PNG/曲线/栅格图/MP4。没有在本机实际跑这三段完整自然视频；本机仅进行了输入预检与封装单元/回归测试。
2026-11-08起默认torch/cuda，后端标签为py_spike_cuda；追加`--device cpu`运行已有NumPy参考。缺少CUDA时明确报错。GPU环境、指标及服务器验收见[GPU_EXECUTION.md](GPU_EXECUTION.md)，本机没有GPU通过或提速结论。下方cpu路径仅为命名示例；默认实际生成cuda标签。

## 命名和登记

实际时间由运行服务器生成的北京时间决定，示例：

```text
outputs/recordings/daria_walk/
  20261108_180000__daria_walk_180x144_f000084__py_spike_cpu__r01/
    artifact.json
    model_config.json
    model_report.json
    monitors/...

outputs/visualizations/daria_walk/
  20261108_180000__daria_walk_180x144_f000084__py_spike_cpu__r01/
    mt_spikes__v01/
      artifact.json
      visualization.json
      mt_spiking.mp4
      mt_exc_rates.png
      mt_inh_rates.png
      mean_rates.png
      spike_raster.png
```

保留原生文件名，不给每个values.f32/mp4重复加长前缀。第二次实验用--repeat 2；参数/显示变化用新运行或新view版本，不能覆盖。
工具按planned→running→completed或failed更新experiments/registry.csv，并在模型/展示目录写artifact.json。completed表示程序和基本完整性检查完成，不代表科学指标passed。
展示artifact关联父运行ID；输入路径、SHA256、神经参数、设备和命令均可追溯。运行中不要并发编辑同一registry；写入使用短锁，锁冲突会明确失败而不是覆盖他人更新。

每个完整视频的默认模型时长：Daria 4.2s、paragliding 3.5s、snowboard65 3.25s；由配套50ms/帧得出，不是原拍摄帧率声明。
默认展示第31帧（零起点30），短试跑自动展示最后一帧；默认E/I色标0..300Hz。绘图参数改变时建立新view_id，直接重用原响应，不必再次仿真。
