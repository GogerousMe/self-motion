# 自运动刺激：当前应采集什么

## 与单方向光栅的区别

单方向光栅检查局部运动编码。自运动刺激需要一个有深度的静态场景，随相机位置与姿态变化生成连续图像。
输入V1的是随机点／场景视频；论文中的箭头是光流可视化，不应把箭头图本身作为视觉输入。

已核对 process/scripts/VisualStimulus/GratingStim.m 与 BaseStim.m：
- GratingStim 的方向定义为0右、90上、180左、270下；BaseStim.save() 会先flipud再写盘，plot也显示flipud结果。
- 新Python刺激使用图像坐标0右、90下，因此 theta_python=(-theta_matlab)%360。
- 光栅速度为tempFreq/spatFreq像素/帧。原默认[0.1,0.1]为1px/frame，不是1.5px/frame。
- 前进／转弯不能通过给整幅光栅换一个运动方向获得；需要三维透视与相机运动。

## 文献依据与差异

Layton等2022使用地面随机点和AirSim拟真环境，研究平移航向、路径曲率、CW/CCW和视线相对切线偏角。
原文方法／Table2给出512×512、30fps、3m/s、90°视场、1.61m高度、2000地面点和1–50m深度等条件。
参考：https://doi.org/10.1088/1748-3190/ac709b 。本实现参照研究范式，未声称逐项复现论文数据集。

当前为配合已有V1工程选择80×80、20fps（50ms/帧）、250点、2–50m深度。不能直接沿用论文精度结论。
相机水平视场90°，方形像素，fx=fy=40px，cx=cy=39.5px，高度1.61m。
每段80帧，显示时长4秒；相机采样时刻0..3.95秒。采集后可先排除前20帧启动瞬态，但应检查是否已稳定。

## 已制作的14段视频

| 名称 | 场景／运动 | 当前用途 |
|---|---|---|
| cloud_static | 静态3D点云，相机不动 | 第一批，静止负对照 |
| cloud_forward | 3D点云，相机正前进 | 第一批，完整视场径向膨胀 |
| ground_forward | 地面点阵，相机正前进 | 第一批，地面航向基线 |
| ground_heading_left10 | 相机朝向不变，速度向左偏10° | 第一批，左侧膨胀中心 |
| ground_heading_right10 | 相机朝向不变，速度向右偏10° | 第一批，右侧膨胀中心 |
| cloud_backward | 3D点云，相机后退 | 收缩模式，后端待扩展 |
| ground_yaw_left/right | 位置固定，原地向左／右转头10°/s | 旋转对照，不能当直线航向 |
| ground_turn_left/right_r45 | 以3m/s沿45m半径圆弧行进，视线沿切线 | 区分左右转；后端待扩展 |
| ground_turn_right_r20/r85 | 右转，不同曲率 | 对比曲率影响 |
| ground_turn_right_r45_gaze_left/right20 | 同一类45m右转，视线相对切线偏-20°／+20° | 对比注视偏角影响 |

偏左直行不是左转：前者相机yaw固定，后者相机位置和yaw随曲线路径共同变化。
原地向右转动相机会使大部分场景向左扫过；“左向光流”不能直接解释成“左转”。
将图片绕图像中心旋转主要模拟roll，不能替代平面车辆转弯的yaw。

第一批5段与当前后端范围最接近。当前MSTd仅有径向膨胀模板和水平航向读出，尚无收缩模板、纯旋转判断或曲率解码器。
其余9段可以先保留，但不能用现有输出角度声称已识别后退／左右转。
半径5m对应3m/s时约34.4°/s的旋转，建议待正常曲率基线完成后再作强转弯测试。

## 实现几何

世界x右、y上、z初始前方；图像x右、y下；yaw正号明确规定为向右。
静态世界点P相对于相机位置C，先按相机yaw转换到Xc,Yc,Zc，再按针孔成像：

```text
u = cx + fx * Xc/Zc
v = cy - fy * Yc/Zc
```

地面世界点y=0，相机高度1.61m，因此点在地平线下方。点云覆盖上下视野。
点在世界中保持固定，只有离开视野／深度范围后才重采样；每次重生分配新ID。
恒定速度v的右／左圆弧以omega=sign*v/R旋转；相机沿切线朝向，额外叠加gaze_offset。
R=45m、v=3m/s时，角速度约3.82°/s。视线偏角与路径旋转分别记录。
随机点采用固定视网膜大小的柔和亮点，不模拟实体物体随距离改变尺寸；这是随机点实验刺激，不是自然场景渲染。
没有动态物体分支；先建立静态世界下的自运动基线。

## 文件与生成

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
python -m mstd generate-selfmotion --output outputs/selfmotion_01 --video
```

视频导出需要PATH中存在ffmpeg；模型和DAT生成无需新增Python依赖。
每个条件包含：
- .avi：FFV1无损灰度视频。
- .dat：可直接供VisualStimulus.readFrameChar读取的同一帧序列。
- .pgm：第一帧预览。
- .truth.jsonl：逐帧相机位置、速度、yaw、角速度、曲率、相对平移方位及稀疏点光流。
- manifest.json与intrinsics.json：参数、输入文件SHA256、每帧时长和待采集响应路径。

稀疏光流du/dv单位为像素/帧，来自同一世界点相邻两帧的精确投影差；最后一帧及下一帧不可见点为空。
静止／纯旋转没有平移方向，heading_camera_deg为null；后退为180°，不能把收缩中心0°误称为前进航向0°。

AVI与DAT逐像素一致。建议直接使用配套DAT送入现有V1，不必再次视频转码。
若使用旧MATLAB PictureStim流程重新转换，须注意BaseStim.save的flipud，确认最终输入与真值方向仍一致。

## Linux采集

你可以继续按原有方法逐个采集，只要保留输入名与对应spk_V1.dat、帧数、50ms/帧和模型版本。
自动化脚本是可选工具。若采用之前说明中的两个环境变量接口，可运行：

```bash
python3 collect_calibration.py --suite /path/selfmotion_suite --executable /path/v1_runner --workdir /path/workdir --runs cloud_static cloud_forward ground_forward ground_heading_left10 ground_heading_right10
```

脚本现在同时接受方向校准和自运动清单；--runs可只采集选定条件。
脚本不提供或改写V1数学模型，也不代替验证实际链接的修改版库。

## 返回响应后

保留完整条件名及对应spk_V1.dat，用现有convert-spikes导入，内参使用本套intrinsics.json。
先检查静止的持续响应、点云／地面膨胀方向和±10°方向偏差，再比较不同投影矩阵。
必须等真实响应到位才评估精度；当前视频与几何测试通过不代表V1–MSTd识别通过。
