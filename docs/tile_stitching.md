# 大图切片去重拼接与实际长度计算

## 目的

4K/8K 桥梁巡检图片直接缩放会丢失细裂缝。切片推理可以保留局部像素密度，但重叠区域会产生重复检测，裂缝跨越切片边界时还会被拆成多段。本模块负责：

1. 按 `tile` 和 `overlap` 生成不重复的切片起点。
2. 将每个切片的掩码映射回原图坐标。
3. 将同类、在重叠区相交的掩码合并为一个物理缺陷实例。
4. 对合并后的完整掩码计算骨架长度、P95 宽度、最大宽度和面积。
5. 使用 `mm_per_pixel` 将像素测量转换为实际尺寸。

## 命令

```powershell
cd D:\zm\newA\a17-bridge-defect
$env:PYTHONPATH = "$PWD\src"

python -m a17.eval.tile_inference `
  --model weights/a17_seg_v2_frozen.pt `
  --image path/to/4k.jpg `
  --tile 1024 `
  --overlap 128 `
  --imgsz 1024 `
  --conf 0.25 `
  --scale 0.42 `
  --device 0 `
  --output outputs/tile_inference
```

也可以直接使用 `predict-seg`，当输入是单张大图时会自动进入切片拼接流程：

```powershell
python -m a17 predict-seg `
  --model weights/a17_seg_v2_frozen.pt `
  --source path/to/4k.jpg `
  --scale 0.42 `
  --tile 1024 `
  --overlap 128
```

## 去重与拼接规则

- 只对相同类别进行合并。
- 两个掩码在重叠区有交集，且交集占较小掩码比例达到阈值时，视为同一缺陷的重复检测。
- 对于 `crack`，两个片段若都呈水平或都呈垂直细长分布、间隔不超过 `bridge_px`，且两边置信度都不低于 `bridge_min_confidence`，则按同一裂缝续接。续接处会绘制一条窄连接线，避免切片边缘留下小断口。
- 合并采用逐实例掩码并集，不改变原图坐标。
- 合并后再次过滤小于 `12 px` 的碎片，避免噪声进入长度统计。
- `source_count > 1` 表示该实例由多个切片预测合并而来。
- `bridge_count > 0` 表示结果使用了端点邻近续接；该数值可通过 JSON 查看。

默认续接参数为：

```text
bridge_px = 24
bridge_min_confidence = 0.10
```

邻近续接只用于较可信的裂缝片段。低置信度检测仍会保留为独立候选，不会因距离接近而强行连接。拼接后的 `connected_components` 应优先为 1；如果仍大于 1，说明结果中还存在无法安全连接的断段。

## 置信度与弱裂缝

Demo 的置信度阈值可在页面中调整，也可通过启动参数设置：

```powershell
python -m a17.demo_server `
  --model weights/a17_seg_v2_frozen.pt `
  --conf 0.10
```

建议先使用 `0.10` 检查弱裂缝召回率，再结合验证集误检情况决定最终阈值。低于 `0.20` 的结果会在页面中标记为“低置信度，必须人工复核”。不要直接用低阈值结果生成最终病害统计。

## 长度计算方法

旧实现直接统计骨架像素个数，这会把 45 度斜线低估约 29%。新实现把骨架视为 8 邻接图：

- 水平/垂直相邻边长度记为 `1 px`。
- 对角相邻边长度记为 `sqrt(2) px`。
- 每条无向边只统计一次，再乘以标定系数。

因此：

```text
length_px = 骨架图所有无向边长度之和
length_mm = length_px * mm_per_pixel
```

宽度使用欧氏距离变换：

```text
width_mm_p95 = 2 * percentile(distance_transform(mask), 95) * mm_per_pixel
width_mm_max = 2 * max(distance_transform(mask)) * mm_per_pixel
```

## 标定要求

`mm_per_pixel` 应来自真实标定，例如已知宽度标尺、镜头内参/高度换算或现场测量。没有标定时只能使用像素指标，不能把 `length_mm` 当作真实长度。
