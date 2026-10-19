# A17 第 9 周技术报告

## 1. 完成范围

本周完成并交付以下内容：

1. YOLO11n-seg P2/4 高分辨率分支与 P2-P5 多尺度 PANet 配置
2. 边界感知分割损失接入
3. V1 第一轮消融实验与消融表
4. 实例后处理、骨架化、长度/宽度量化 V1
5. 量化样例、像素比例标定 V2、量化参数 V2
6. V2 生产模型冻结、FP16 ONNX 导出
7. 批量并行切片推理、4K 性能基准与速度报告

## 2. 模型结构

### 2.1 P2 分支

新增：

```text
configs/models/a17_yolo11n_seg_p2.yaml
```

该模型保留原始 P3/P4/P5 neck，额外从 P3 neck 上采样并与 backbone P2 拼接，得到 P2/4 高分辨率检测层。`Segment` 的四个输入尺度为 P2/P3/P4/P5。

### 2.2 多尺度 PANet

新增：

```text
configs/models/a17_yolo11n_seg_p2_ms.yaml
```

该模型加入完整双向特征金字塔：

```text
Top-down:  P5 -> P4 -> P3 -> P2
Bottom-up: P2 -> P3 -> P4 -> P5
```

四个尺度均接入分割头，stride 为：

```text
4 / 8 / 16 / 32
```

### 2.3 边界损失

边界损失实现在：

```text
src/a17/train/boundary_loss.py
```

统一训练入口已支持：

```powershell
python -m a17.train.train_yolo `
  --model configs/models/a17_yolo11n_seg_p2_ms.yaml `
  --data data/processed/dacl10k_subset/dataset.yaml `
  --boundary-weight 0.3
```

损失在标准分割 BCE 外增加预测边界与真实边界的 Sobel 边缘 L1 项，用于强化薄裂缝边界。

## 3. V1 第一轮消融

消融数据：`dacl10k_subset`，239 张训练图、40 张验证图。

设置：`imgsz=512`、`batch=4`、12 epochs、AdamW。

| 模型 | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 | 推理 ms | FPS |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.1143 | 0.0417 | 0.0895 | 0.0340 | 3.367 | 297.0 |
| +P2 | 0.0039 | 0.00087 | 0.0027 | 0.00076 | 7.057 | 141.7 |
| +P2+多尺度 | 0.0101 | 0.00268 | 0.0096 | 0.00399 | 6.251 | 160.0 |
| +P2+多尺度+边界损失 | 0.0101 | 0.00268 | 0.0096 | 0.00399 | 4.100 | 243.9 |

结论：

- 在短周期子集训练预算下，Baseline 仍显著优于新增结构分支。
- P2/多尺度分支的可训练参数和原型分支从通用权重迁移不足；12 epochs 不足以收敛。
- 边界损失可运行，但第一轮不能单独弥补结构分支的欠训练。
- V1 保留为实验模型：

```text
weights/a17_seg_v1_experimental.pt
configs/model_v1_ablation.json
```

后续建议从兼容的 3 类分割权重 warm-start，或在完整数据上延长训练周期，再决定是否将 P2/多尺度纳入生产模型。

## 4. 量化模块 V1

实现：

```text
src/a17/quantification_v1.py
```

功能：

- 小面积碎片过滤
- 可选闭运算和孔洞填充
- Mask 骨架化
- 端点、分叉点统计
- 8 邻接骨架长度
- P95 宽度、最大宽度、面积
- 毫米单位转换

真实预测样例结果：

| 指标 | 数值 |
|---|---:|
| 长度 | 163.977 mm |
| P95 宽度 | 8.566 mm |
| 最大宽度 | 12.600 mm |
| 面积 | 1307.477 mm² |
| 骨架像素 | 351 |
| 分叉点 | 0 |

样例文件：

```text
outputs/quantification_v1_sample/mask_v1.png
outputs/quantification_v1_sample/skeleton_v1.png
outputs/quantification_v1_sample/quantification_v1.json
```

## 5. 标定与量化参数 V2

标定工具：

```text
src/a17/calibrate.py
```

当前项目使用的是临时标定：

```text
configs/calibration_v2.json
mm_per_pixel = 0.42
provisional = true
```

该数值来自项目现有示例参数，正式部署前必须用现场标尺、双点测量或相机几何换算替换。

量化参数：

```text
configs/quantification_v2.json
```

定义了长度方法、P95 宽度、最小面积、Mask/骨架保存等参数。

## 6. 冻结模型 V2

生产模型冻结为：

```text
weights/a17_seg_v2_frozen.pt
```

来源：`weights/a17_seg_finetune_v5_last.pt`

冻结清单：

```text
configs/model_v2_freeze.json
```

V2 独立验证结果：

| 指标 | 数值 |
|---|---:|
| Mask mAP50 | 0.1820 |
| Mask mAP50-95 | 0.0695 |
| Box mAP50 | 0.2356 |
| Box mAP50-95 | 0.1212 |
| 推理时间 | 5.84 ms |
| FPS | 171.3 |

## 7. FP16 ONNX 部署

导出文件：

```text
weights/a17_seg_v2_frozen.onnx
```

导出参数：

- 输入：`1 x 3 x 640 x 640`
- 输入类型：`float16`
- 输出：`(1, 39, 8400)` 和 `(1, 32, 160, 160)`
- 文件大小：约 5.6 MB

ONNX Runtime 检查时 CUDA Provider 因本机缺少对应 cuDNN/CUDA 运行库而回退 CPU：

```text
CPU: 17.374 ms, 57.56 FPS (640)
```

PyTorch CUDA FP16 推理已验证可用，4K 性能测试使用该路径。

## 8. 并行切块与 4K 性能

并行切块实现：

```text
src/a17/eval/parallel_tile.py
```

4K 测试图像：

```text
4128 x 3096
```

切片参数：

```text
tile=1024, overlap=128, imgsz=1024
```

最终结果：

| 精度 | batch | workers | 平均 ms | P50 ms | P95 ms | 全图 FPS |
|---|---:|---:|---:|---:|---:|---:|
| FP16 | 4 | 1 | 771.77 | 760.66 | 794.99 | 1.296 |
| FP32 | 4 | 1 | 825.53 | 829.29 | 832.13 | 1.211 |

结论：

- 批量切片 `batch_size=4` 明显优于逐片推理。
- 额外 Python 线程 workers>1 会因 GPU 竞争变慢，不建议作为主路径。
- FP16 相比 FP32 在当前 4K 端到端切片测试中约提升 6.5%。

完整速度报告：

```text
outputs/benchmark_4k_final/speed_benchmark.json
outputs/benchmark_4k_fp32_final/speed_benchmark.json
outputs/benchmark_onnx/benchmark.json
```

## 9. 风险与下一步

1. `calibration_v2.json` 当前为临时标定，不能直接作为工程验收尺寸。
2. V1 的 P2/多尺度结构首轮未超过 Baseline，需要更长训练和更好的初始化策略。
3. ONNX Runtime CUDA Provider 需要安装与 CUDA/cuDNN 匹配的版本；当前 FP16 ONNX 的 GPU 加速验证使用 PyTorch 路径完成。
4. 下一步建议：
   - 使用完整 `a17_combined` 数据对 V1 结构进行 50 epochs 以上训练；
   - 用真实标定替换 `0.42 mm/px`；
   - 在部署机器上补装 TensorRT/cuDNN，导出 engine 并复测 4K 端到端延迟。
