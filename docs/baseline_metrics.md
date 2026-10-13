# 基线指标表

## 检测基线 GYU-DET（YOLO11n-detect）

| 数据 | 指标 | 数值 |
|---|---|---|
| GYU-DET 子集 | box mAP50 | 0.061 |
| GYU-DET 子集 | box mAP50-95 | 0.022 |
| GYU-DET 全量 | box mAP50 | 0.343 |
| GYU-DET 全量 v2（60 epochs） | box mAP50 | 0.448 |
| GYU-DET 全量 v2（60 epochs） | box mAP50-95 | 0.257 |
| GYU-DET 全量 v1 | box mAP50-95 | 0.182 |

## 分割基线 dacl10k 快速子集（YOLO11n-seg）

模型：runs/segment/outputs/train/dacl_seg_baseline/weights/best.pt

| 指标 | 数值 |
|---|---|
| box mAP50 | 0.148 |
| box mAP50-95 | 0.0547 |
| mask mAP50 | 0.163 |
| mask mAP50-95 | 0.0644 |
| FPS | 约 135 |

## 分割基线合并完整数据集（YOLO11n-seg）

模型：runs/segment/outputs/train/dacl_seg_combined/weights/best.pt
数据：data/processed/a17_combined（train 7114 / val 979）

总体指标：

| 指标 | 数值 |
|---|---|
| box mAP50 | 0.215 |
| box mAP50-95 | 0.112 |
| mask mAP50 | 0.166 |
| mask mAP50-95 | 0.0599 |
| 推理耗时 | 2.4 ms/图 |
| 预处理 | 0.6 ms/图 |
| 后处理 | 1.2 ms/图 |
| FPS | 约 413 |

分类指标：

| 类别 | Mask mAP50 | Mask mAP50-95 |
|---|---|---|
| crack | 0.072 | 0.0195 |
| spalling | 0.166 | 0.0703 |
| exposed_rebar | 0.260 | 0.0898 |

## 分割基线高分辨率长训练（YOLO11n-seg, imgsz=1024, 40 epochs）

模型：runs/segment/outputs/train/dacl_seg_highres_long/weights/best.pt

| 指标 | 数值 |
|---|---|
| box mAP50 | 0.233 |
| box mAP50-95 | 0.118 |
| mask mAP50 | 0.177 |
| mask mAP50-95 | 0.0679 |
| FPS | 约 164 |

分类指标：

| 类别 | Mask mAP50 | Mask mAP50-95 |
|---|---|---|
| crack | 0.069 | 0.0209 |
| spalling | 0.183 | 0.0765 |
| exposed_rebar | 0.279 | 0.106 |

## 分割基线边界损失微调（BoundarySegmentationLoss, imgsz=640, 10 epochs）

模型：runs/segment/outputs/train/dacl_seg_boundary_ft/weights/best.pt
起点：dacl_seg_highres_long best.pt；boundary_weight=0.3

| 指标 | 数值 |
|---|---|
| box mAP50 | 0.229 |
| box mAP50-95 | 0.116 |
| mask mAP50 | 0.175 |
| mask mAP50-95 | 0.0640 |
| FPS | 约 417 |

分类指标：

| 类别 | Mask mAP50 | Mask mAP50-95 |
|---|---|---|
| crack | 0.082 | 0.0250 |
| spalling | 0.177 | 0.0736 |
| exposed_rebar | 0.266 | 0.0932 |

## 说明

- mask mAP50 即 mask IoU 阈值为 0.5 时的平均精度，是分割评估的标准口径。
- 加入 CrackSeg9k 和 Concrete Crack Segmentation 后，crack 的 Mask mAP50 从 0.009 提升到 0.072；但裂缝仍是瓶颈，下一步需增加边界损失、高分辨率输入和更长训练。
- SDNET2018 只作为背景负样本，参与训练但不含正实例。
- FPS 在 RTX 5060 Laptop GPU、imgsz=640、batch=8 下测得，不是 4K 输入；4K 的 200ms 指标需在切块 + TensorRT 阶段单独验证。
