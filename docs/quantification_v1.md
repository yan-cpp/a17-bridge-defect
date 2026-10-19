# 缺陷量化模块 V1

## 模块范围

量化模块 V1 位于：

```text
src/a17/quantification_v1.py
```

它接收实例 Mask，执行：

1. 小面积碎片过滤
2. 可选形态学闭运算
3. Mask 骨架化
4. 端点和分叉点统计
5. 中心线长度计算
6. P95 宽度、最大宽度和面积计算
7. 毫米单位转换

## 长度算法

骨架按 8 邻接图统计无向边：

```text
水平/垂直边 = 1 px
对角边 = sqrt(2) px
length_mm = length_px * mm_per_pixel
```

## 宽度算法

```text
distance = distance_transform_edt(mask)
width_px = 2 * distance
width_mm_p95 = percentile(width_px, 95) * mm_per_pixel
width_mm_max = max(width_px) * mm_per_pixel
```

## 命令行

```powershell
cd D:\zm\newA\a17-bridge-defect
$env:PYTHONPATH = "$PWD\src"

python -m a17.quantification_v1 `
  --mask path/to/instance_mask.png `
  --scale 0.42 `
  --class-key crack `
  --output outputs/quantification_v1
```

输出文件：

- `mask_v1.png`：后处理后的实例 Mask
- `skeleton_v1.png`：骨架
- `quantification_v1.json`：长度、宽度、面积、端点、分叉点等参数

## 标定说明

`mm_per_pixel` 必须来自实际标定文件。没有标定时只能使用像素指标，不能把毫米结果当作真实尺寸。
