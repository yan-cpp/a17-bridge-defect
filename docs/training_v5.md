# A17 分割模型微调记录 v5

## 训练设置

- 基础权重：`runs/segment/outputs/train/dacl_seg_highres_long/weights/best.pt`
- 数据：`data/processed/a17_combined/dataset.yaml`
- 输入尺寸：`1024`
- batch：`2`
- epochs：`12`
- optimizer：`AdamW`
- 初始学习率：`5e-05`
- warmup：`1 epoch`，`warmup_bias_lr=0.0`
- mosaic / mixup / copy-paste：全部关闭
- 训练结果：`runs/segment/outputs/train/dacl_seg_finetune_v5`

## 指标对比

统一使用 `imgsz=1024`、`split=val`、`batch=2` 评估。

| 指标 | 原最佳 highres_long | 微调 v5 last | 变化 |
|---|---:|---:|---:|
| Mask mAP50 | 0.1769 | 0.1820 | +0.0051 |
| Mask mAP50-95 | 0.0678 | 0.0695 | +0.0017 |
| Box mAP50 | 0.2324 | 0.2356 | +0.0032 |
| Box mAP50-95 | 0.1179 | 0.1212 | +0.0033 |

分类 Mask mAP50：

| 类别 | 原最佳 highres_long | 微调 v5 last |
|---|---:|---:|
| crack | 0.0694 | 0.0720 |
| spalling | 0.1830 | 0.1870 |
| exposed_rebar | 0.2790 | 0.2870 |

## 部署权重

最终复制为：

```text
weights/a17_seg_finetune_v5_last.pt
```

兼容的对比报告：

- `outputs/eval/base_highres_1024_metrics.json`
- `outputs/eval/dacl_seg_finetune_v5_last_metrics.json`
- `outputs/eval/dacl_seg_finetune_v5_metrics.json`

## 说明

第一次尝试使用较高学习率并保留默认 `warmup_bias_lr=0.1`，第一个 epoch 后 Mask mAP50 降到约 0.13。原因是微调时分类/回归头部 bias 被 warmup 大幅扰动。v5 将 warmup bias 学习率设为 0，并关闭强 mosaic 类增强后，精度稳定超过原最佳模型。
