# A17 物理量化模块

## 定位

量化模块从每个病害实例的 Mask 计算物理长度、最大宽度和面积。量化不改变检测或分割结果，只在后处理阶段使用。

## 输入输出

输入：

- 二值实例 Mask（PNG，0/255 或 0/1）。
- `scale_mm_per_pixel`：毫米每像素，由相机标定或赛题转换系数提供。

输出字段：

- `length_mm`：骨架化后的裂缝长度。
- `width_mm_max`：距离变换最大宽度的 2 倍。
- `width_mm_p95`：距离变换 P95 宽度的 2 倍，作为稳健结果。
- `area_mm2`：Mask 面积换算后的物理面积。
- `skeleton_px`、`connected_components`：用于审计和排错。

## 算法

1. 对 Mask 二值化。
2. 使用 skimage 对每个连通域做骨架化，统计骨架像素数。
3. 长度 = 骨架像素数 × mm_per_pixel。
4. 使用 scipy 距离变换估计局部半径。
5. 最大宽度 = 2 × max(distance)。
6. P95 宽度 = 2 × percentile(distance, 95)。

## 命令

```powershell
cd D:\zm\newA\a17-bridge-defect
$env:PYTHONPATH = "$PWD\src"
python -m a17 quantify-mask --mask path/to/mask.png --scale 0.42
python -m a17 predict-seg --model runs/segment/outputs/train/dacl_seg_combined/weights/best.pt --source path/to/image.jpg --imgsz 1024 --scale 0.42
```

## 注意

- 无标定参数时无法得到可靠的物理尺寸，必须记录标定来源。
- 骨架像素法对 45 度走向的裂缝会低估长度，后续可改用图最短路径或方向加权。
- 距离变换宽度对断点和噪声敏感，正式结果建议使用 P95，而非单点最大值。
