# Metrics And Measurement Rules

## Detection

必须按类别报告：

- mAP@0.5
- mAP@0.5:0.95
- Precision
- Recall
- 漏检数量和误检数量

验收门槛：所选模型不低于YOLO或Transformer对照基线。

## Segmentation

必须按类别报告：

- Mask IoU
- Dice
- Boundary F1
- Mask连通域断裂情况

验收门槛：Mask IoU不低于所选基线。

## Quantification

### Pixel Scale

优先使用赛题提供的像素-物理尺寸转换系数。若需要从相机参数计算：

```text
mm_per_pixel = flight_height_mm * sensor_pixel_size_mm / focal_length_mm
```

在透视和倾斜拍摄条件下，应增加单应校正或明确限制测量场景。

### Length

1. 对实例Mask做骨架化。
2. 去除短分支和孤立点。
3. 在骨架图中寻找最长路径。
4. 累计路径长度并乘以 `mm_per_pixel`。

### Width

1. 沿骨架提取正交截面。
2. 使用距离变换或截面长度估算局部宽度。
3. 过滤异常值。
4. 输出P95宽度、中位宽度和最大值，正式结果使用P95。

### Error Metrics

- 长度MAE和MAPE。
- 宽度MAE和MAPE。
- 重复计算波动率。
- 按病害类别、距离、光照和遮挡条件分组统计。

内部目标：

- 长度MAPE不超过10%。
- 宽度MAPE不超过15%。
- 重复计算波动率不超过5%。

## Latency

必须报告以下分项：

- `preprocess_ms`
- `inference_ms`
- `postprocess_ms`
- `quantification_ms`
- `total_ms`

在3840x2160输入、RTX4090级别GPU上，端到端P95目标不超过200ms。
