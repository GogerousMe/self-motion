# 下一轮刺激：纹理校准、静态对照、连续切换与自运动

日期：2026-10-25。所有新预览为MP4/H.264，未生成新的AVI；原目录和原DAT不覆盖。

## 原始目录是什么

`outputs/calibration_texture_01/` 含8个独立DAT：texture_d000、d045、d090、d135、d180、d225、d270、d315，以及manifest和PGM预览。每段80×80、80帧、50ms/帧、4秒、速度1.5px/frame。它们是固定随机亮点纹理的平移，不是三维点云，也不是每帧重新随机生成。

原8段只有移动纹理；没有专门的静止纹理负对照，也没有连续保持纹理位置的方向切换。因为独立片段从初始纹理位置开始，简单拼接会重置图像相位，不能宣称测试了无重置方向变化。

`outputs/selfmotion_01/` 含14类相机运动刺激及逐帧真值、内参和原始视频；当前只合并5个适合膨胀模板基线的条件。后退、纯旋转、曲线等原文件保留，暂不混入当前验收。

## 已生成两套连续输入

### A. 纹理综合测试

路径：`outputs/validation_sequences_01/texture/`

| 文件 | 用途 |
|---|---|
| all_stimuli.dat | 唯一连续CARLsim图像输入 |
| manifest.json、segments.csv | 文件校验、帧数、条件、方向和分段时间 |
| preview/all_stimuli.mp4 | 全90秒带注释输入预览 |
| preview/all_stimuli_segment_*.png | 各段截图 |

80×80、单通道、1800帧、50ms/帧，共90000ms，DAT大小11520025字节。方向按0右、45右下、90下、135左下、180左、225左上、270上、315右上。

| 时间 | 内容 | 目的 |
|---|---|---|
| 0～4秒 | 固定初始纹理 | 有纹理但无运动的负对照 |
| 4～52秒 | 原八方向按顺序各4秒，每段后保持末帧静止2秒 | 保留原片段内容并观察停止运动后的残留 |
| 52～54秒 | 复位到初始纹理，静止2秒 | 明确隔开图像复位与后续连续测试 |
| 54～86秒 | 八方向依次各4秒，始终使用同一纹理位置状态 | 无黑场、无相位复位的方向切换 |
| 86～90秒 | 保持最后纹理不动 | 连续运动停止后的背景误报与消退 |

三类测试均包含在这一份DAT里，不是额外要求三个文件。整个DAT没有插入全黑帧。静态仍然有真实纹理，不能给静态段标运动方向。

八个原移动片段的像素逐字节保留。真正连续切换只指54～86秒那一块；原片段之间的场景复位有明确标签，不伪装成平滑运动。切换边界帧属于瞬态，应单独看，不应拿稳定期均值代替跟踪延迟。

### B. 自运动基础测试

路径：`outputs/validation_sequences_01/selfmotion/`

80×80、单通道、400帧、50ms/帧，共20000ms，DAT大小2560025字节。

| 时间 | 条件 | 当前检查目标 |
|---|---|---|
| 0～4秒 | cloud_static | 静止时不输出有效平移航向 |
| 4～8秒 | cloud_forward | 全视场扩张，航向约0° |
| 8～12秒 | ground_forward | 地面扩张，航向约0° |
| 12～16秒 | ground_heading_left10 | 相机朝向固定，左偏10°平移 |
| 16～20秒 | ground_heading_right10 | 相机朝向固定，右偏10°平移 |

同样含 `all_stimuli.dat`、`segments.csv`、`manifest.json`，当前自运动预览使用 `preview02/all_stimuli.mp4`（包含正确的相机米制速度标注）。另有 `intrinsics.json` 和合并 `truth.jsonl`；真值明确记录条件名、全局帧号以及原条件的局部时间/相机真值。

这是5个独立场景/相机条件的汇编，不是一条连续真实相机轨迹。每4秒边界可能有图像和相机状态跳变。V1采集整份DAT时持续运行，后续分析按分段隔离，MSTd在独立条件边界重置，并检查启动瞬态。若残留影响基线，需用保留的原独立DAT重新复核；不把场景切换当作真实转向。

## 三维点云从哪里来、给谁使用

当前三维世界由 `src/mstd/selfmotion.py` 构造：随机点位于设定深度范围的虚拟空间，或位于地面；相机有已知内参、位置和朝向。使用针孔投影，将每个时刻可见世界点渲染为80×80二维灰度图。

```text
程序生成静态三维世界点 + 指定相机运动
                  ↓ 透视投影、渲染
            二维图像序列 DAT → V1 → MT → MSTd → 航向
                  ↓                         ↓
             已知运动真值 ─────────────── 比较误差
```

DAT里没有附带XYZ深度通道。图像中不同点的相对位移包含了由三维几何产生的透视运动关系，MT编码这些二维运动，MSTd整合它们估计航向。当前不是先从V1/MT恢复三维点云再送给MSTd，也不需要购买或采集激光点云。

当前已有生成器采用250个点、2～50m深度、90°水平视场；地面相机高度1.61m。生成时的相机参数和真值用于核验，不作为MSTd的隐藏答案。自然视频也可作为未来输入，但测试时需相应标定和独立运动真值。

左右偏平移不是左右转。后续旋转/曲率解码需要扩展模型，不因已生成这些图像而自动实现。

## 在服务器做什么

1. 将 `outputs/validation_sequences_01/texture/` 和 `selfmotion/` 分别上传，保留DAT、CSV和manifest配套；自运动再保留内参和真值。预览MP4可只放本机，不必送入V1。
2. 分别运行两次CARLsim：纹理1800帧/90000ms，自运动400帧/20000ms，每帧50ms。
3. 每次完整实验开始前重置V1，整份DAT内部保持连续，不在每段重置。确认输入80×80、V1输出深度84；每份响应单独命名或单独目录保存。
4. 保存运行日志和实际输入哈希，不能凭最后一次脉冲确定实验总时长。
5. 后续纹理分析使用这次的新manifest；不能继续使用旧光栅清单。

纹理序列已接入现有 `analyze_mt_gratings.py`、`visualize_mt_gratings.py`、`compare_mt_stages.py` 的清单读取。请同步这些更新脚本；静态误报仍须另看，`moving_directions_passed`不包含静态验收。自运动清单不能送入方向校准脚本，因为“相机航向”与“全图同向运动”语义不同；需要按五条件和真值进行MSTd分析。

## 重新生成和单独预览脚本

重新生成（新输出目录，不覆盖）：

```bash
export PYTHONPATH="$PWD/src"
python examples/prepare_validation_sequences.py \
  --texture-suite outputs/calibration_texture_01 \
  --selfmotion-suite outputs/selfmotion_01 \
  --output outputs/validation_sequences_02
```

预览任意原始DAT或整个原目录，每个DAT生成一个MP4：

```bash
python examples/preview_stimulus_dat.py \
  --input outputs/calibration_texture_01 \
  --output outputs/texture_original_mp4_01

python examples/preview_stimulus_dat.py \
  --input outputs/selfmotion_01 \
  --output outputs/selfmotion_original_mp4_01
```

预览合并DAT：

```bash
python examples/preview_stimulus_dat.py \
  --input outputs/validation_sequences_01/texture/all_stimuli.dat \
  --output outputs/texture_combined_mp4_02
```

服务器已移动DAT时可用 `--manifest /实际配套清单路径/manifest.json`。若没有清单，需要显式 `--frame-ms 50`，画面仅标文件名，不猜测方向；清单存在时仍验证哈希与时间。支持 `--max-frames` 和 `--playback-rate 2`，只影响预览。输入DAT不包含任何文字或箭头，标注只在MP4里。

脚本依赖Python>=3.10、mstd源码、同目录 `visualize_mt_gratings.py` 和ffmpeg，无额外Python库。Windows PowerShell改用 `$env:PYTHONPATH = Join-Path (Get-Location) 'src'`，不要直接运行Bash的export。

验证：36项测试通过，覆盖原片段逐字节保留、静态保持、连续位置传递、单个DAT头、五条件选择和真值局部坐标。已核验两份完整MP4为H.264、20fps、分别1800帧/90秒与400帧/20秒；真实神经模型响应须采集后再评价。
