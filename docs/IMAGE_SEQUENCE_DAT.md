# 图片文件夹生成DAT与真实场景测试

`examples/images_to_dat.py`按文件名的自然数字顺序读取JPG/PNG/BMP/TIFF，例如`in1、in2、in10`。一次读取一张图片，写入与CARLsim VisualStimulus兼容的25字节头及uint8帧；type=5（MOVIE_STIM），灰度为TYX，RGB为TYXC逐像素交错。不改变原图片，不覆盖已有DAT或同名JSON；尺寸不一致时默认报错，须明确选择缩放/补边。

运行依赖现有项目及Pillow，可在同一环境执行`python -m pip install Pillow`，或安装项目的`images`可选依赖。同步`examples/images_to_dat.py`与`src/mstd/image_sequence.py`；`pyproject.toml`仅新增可选依赖声明。脚本会定位同项目的src，无需创建新环境。

## 已检查的daria_walk

本机`D:/A_Gorger/program/Datasets/daria_walk`有84张RGB图片，尺寸全部为宽180×高144，名称从`in00001.jpg`到`in00084.jpg`。已经实际生成：

- `input/stimuli/natural/daria_walk/daria_walk__180x144__f000084__dt50ms__gray_native__v01/daria_walk__180x144__f000084__dt50ms__gray_native__v01.dat`：保持原尺寸，84帧灰度。
- `input/stimuli/natural/daria_walk/daria_walk__188x144__f000084__dt50ms__gray_pad__v01/daria_walk__188x144__f000084__dt50ms__gray_pad__v01.dat`：左右各补4列黑色，84帧灰度。
- 各自的`.dat.json`记录顺序、原文件SHA256、DAT哈希、几何变换和设定帧间隔。

本机重建原尺寸版本（PowerShell，一行）：

```powershell
python examples/images_to_dat.py --input "D:\A_Gorger\program\Datasets\daria_walk" --output input/stimuli/natural/daria_walk/daria_walk__180x144__f000084__dt50ms__gray_native__v01/daria_walk__180x144__f000084__dt50ms__gray_native__v01.dat --mode gray --frame-ms 50
```

输出已存在时，使用新文件名。若直接把已生成的DAT及JSON上传到服务器，可以跳过图片转换，服务器也无需为此安装Pillow。

## 可选尺寸和颜色

默认`--geometry native`，不自动缩放。显式变换例子：

```bash
# 保持原像素尺度，180x144左右各补4列，得到188x144
python examples/images_to_dat.py --input /实际图片目录/daria_walk \
  --output experiments/scratch/manual__stimuli__r01/daria_walk_pad_02.dat \
  --mode gray --frame-ms 50 --geometry pad --size 188 144

# 直接缩放为188x144，使用双线性插值（横向比例会改变）
python examples/images_to_dat.py --input /实际图片目录/daria_walk \
  --output experiments/scratch/manual__stimuli__r01/daria_walk_resize_01.dat \
  --mode gray --frame-ms 50 --geometry resize --size 188 144
```

`--mode rgb`保留三通道RGB；默认`gray`使用Pillow的L灰度转换。透明像素先合成到黑色，保留文件中的像素坐标，不依据EXIF自动旋转。`--pattern '*.jpg'`限制文件类型，`--max-frames 20`转换前20张。输入原生帧率已知时，可用`--fps 25`代替`--frame-ms 40`，二者不能同时指定。

DAT本身不储存FPS。当前生成样例用50ms/帧，这是明确设置的实验时间，并非从daria_walk文件名推断的原生帧率。模型配置`frame_ms`必须与生成说明一致；更改帧率时一并更改。逐帧数据始终保持原始顺序，不做插帧。

## GPU运行及视频

将原尺寸DAT及JSON按相对路径上传到服务器项目。继续使用已验证可用的GPU环境，运行整段84帧（无需沿用光栅实验的4800帧参数）：

```bash
conda activate mstd
cd /data/workspace/wyq/mstd/code
export PYTHONPATH="$PWD/src"

python run_model.py --config configs/model_mt_video.json \
  --stimulus input/stimuli/natural/daria_walk/daria_walk__180x144__f000084__dt50ms__gray_native__v01/daria_walk__180x144__f000084__dt50ms__gray_native__v01.dat \
  --output outputs/recordings/daria_walk/legacy__daria_walk_180x144_f000084__py_rate_cuda__r01 --backend torch --device cuda

python examples/compare_mt_stages.py \
  --recording outputs/recordings/daria_walk/legacy__daria_walk_180x144_f000084__py_rate_cuda__r01 \
  --stimulus input/stimuli/natural/daria_walk/daria_walk__180x144__f000084__dt50ms__gray_native__v01/daria_walk__180x144__f000084__dt50ms__gray_native__v01.dat \
  --rate-max 300 --activity-max 1 --colormap jet \
  --progress-every 20 --output outputs/visualizations/daria_walk/legacy__daria_walk_180x144_f000084__py_rate_cuda__r01/mt_stages__v01
```

查看`outputs/visualizations/daria_walk/legacy__daria_walk_180x144_f000084__py_rate_cuda__r01/mt_stages__v01/mt_stages.mp4`。该真实视频不使用光栅的分段manifest，`.dat.json`只提供来源/排序/时间审计，不伪造全图单一方向真值。可换用188×144补边DAT并更换输出目录，其余模型/视频代码不变。与CARLsim对比时，两个程序使用同一份DAT、相同帧间隔及起始状态；不要用旧80×80刺激的脉冲结果直接比较新180×144输入。

## 宽高与响应数量

统一模型从DAT头读取W、H，V1输出`[3,28,H,W]`，每个MT阶段输出`[8,H,W]`。不用另改连接矩阵或配置中的宽高，不要求8或16的整数倍；本版本宽高各至少19，一个序列内保持固定。更大尺寸的实际可用性受显存、内存、时间和监视器文件大小约束，不能视为无限分辨率。

| 空间网格W×H | 像素位置 | V1响应单元84×W×H | 每个MT阶段8×W×H |
| --- | ---: | ---: | ---: |
| 180×144 | 25,920 | 2,177,280 | 207,360 |
| 188×144 | 27,072 | 2,274,048 | 216,576 |

这些是当前连续模型的响应阵列数量；MT投影/池化/归一化是三个阶段，不代表新增三个独立脉冲脑区。180×144输入不能只改DAT头或单独修改网格来冒充188×144，需要明确缩放或补边。补边保持内容像素位移，但会改变边界上下文；缩放会改变像素速度和物体尺度。滤波核、池化半径均以像素计，改变分辨率不保证响应自动等比例一致。后续启用带相机内参的MSTd时，缩放/补边还需同步内参。

## 本机验证

新增4项测试检查数字排序、灰度/RGB布局、固定几何、补边、透明像素、文件长度及哈希、拒绝覆盖、混合尺寸/损坏图片失败清理和CLI入口。完整56项测试中55项通过，1项因没有PyTorch跳过。实际84张图片的两个尺寸DAT均已生成；180×144完整CPU链路、188×144短CPU链路另作运行检查。服务器GPU和真实场景方向响应仍由实际实验验收。
