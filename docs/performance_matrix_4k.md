# A17 4K 性能矩阵与 TensorRT 验证

## 结论

- TensorRT 10.13.3.9 已在 RTX 5060 Laptop GPU、CUDA 12.8 驱动上完成真实 engine 构建、加载和推理。
- 静态 FP16 batch=4 engine 与 PyTorch FP16 batch=4 的检测结果一致：F1 1.0、平均匹配框 IoU 0.9988、最大置信度偏差 0.0006。
- 静态 engine 的 4K 端到端延迟为 1444.78 ms，P50 为 1448.55 ms，仅比本轮 PyTorch FP16 的 1487.05 ms 快约 2.8%。
- 200 ms 验收目标在当前 RTX 5060 Laptop GPU 上尚未达到，不能把当前结果表述为满足实时 4K 指标。

## 环境与输入

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB |
| 驱动 / CUDA | 572.93 / CUDA 12.8 |
| Python / PyTorch | 3.13.14 / 2.11.0+cu128 |
| TensorRT | 10.13.3.9 |
| 输入图片 | `4128 x 3096` |
| 切片 | `tile=1024`, `overlap=128`, `imgsz=1024`, 20 tiles |
| 测量 | 每组预热后重复，端到端包含切片、推理、结果转换与拼接 |

## 完整矩阵

| 后端 | 精度 | batch | 平均 ms | P50 ms | P95 ms | 全图 FPS |
|---|---|---:|---:|---:|---:|---:|
| PyTorch | FP32 | 1 | 1741.59 | 1736.51 | 1782.46 | 0.574 |
| PyTorch | FP32 | 2 | 1598.37 | 1609.88 | 1654.45 | 0.626 |
| PyTorch | FP32 | 4 | 1549.39 | 1533.64 | 1592.05 | 0.645 |
| PyTorch | FP16 | 1 | 1694.95 | 1689.52 | 1733.38 | 0.590 |
| PyTorch | FP16 | 2 | 1552.88 | 1566.65 | 1591.39 | 0.644 |
| PyTorch | FP16 | 4 | 986.60 | 823.84 | 1392.38 | 1.014 |
| TensorRT dynamic | FP16 | 1 | 1390.12 | 1456.19 | 1475.85 | 0.719 |
| TensorRT dynamic | FP16 | 2 | 1415.64 | 1446.37 | 1530.66 | 0.706 |
| TensorRT dynamic | FP16 | 4 | 1506.06 | 1498.08 | 1557.90 | 0.664 |

动态 engine 为 batch 1–4、空间尺寸 32–2048 的通用 profile。其编译耗时约 549 秒，执行时分配约 815 MiB GPU 工作区。动态 shape 没有带来 batch 扩展收益，反而比静态 batch=4 engine 更慢。

## 静态生产引擎对照

| 后端 | 精度 | batch | 平均 ms | P50 ms | P95 ms | 全图 FPS |
|---|---|---:|---:|---:|---:|---:|
| PyTorch | FP32 | 4 | 1554.32 | 1559.68 | 1587.19 | 0.643 |
| PyTorch | FP16 | 4 | 1487.05 | 1494.33 | 1562.65 | 0.672 |
| TensorRT static | FP16 | 4 | 1444.78 | 1448.55 | 1524.23 | 0.692 |

静态 engine 编译耗时约 593 秒，执行时只分配约 113 MiB GPU 工作区，适合固定 `imgsz=1024, batch=4` 的生产路径。

## 校验边界

- 一致性校验使用同类目标、IoU >= 0.5 的贪心框匹配。
- 本次 4K 测试图片只产生 3 个检出，能证明 TensorRT 链路和输出格式正确，不足以单独代表完整验证集的统计精度。
- TensorRT engine 与 GPU、驱动、TensorRT 版本绑定；换机器后必须重新构建，不能直接复制到不同 GPU 作为可移植模型。
- TensorRT 只证明推理后端可运行且输出一致，不替代完整数据集的 Mask mAP 评估。

## 复现

```powershell
cd D:\zm\newA\a17-bridge-defect
python -m pip install -r requirements-tensorrt.txt
powershell -ExecutionPolicy Bypass -File scripts\run_4k_performance_matrix.ps1
```

静态生产 engine 对照命令：

```powershell
python -m a17.deploy --model weights\a17_seg_v2_frozen.pt `
  --format engine --imgsz 1024 --half --batch 4 --workspace 2 `
  --output weights\a17_seg_v2_frozen.engine --device 0

python -m a17.eval.benchmark_matrix `
  --pytorch-model weights\a17_seg_v2_frozen.pt `
  --engine-model weights\a17_seg_v2_frozen.engine `
  --image data\processed\a17_combined\images\val\dacl10k_v2_validation_0545.jpg `
  --tile 1024 --overlap 128 --imgsz 1024 --batch-sizes 4 --repeat 10 `
  --device 0 --output outputs\benchmark_4k_matrix_static_b4
```

结果文件：

```text
outputs/benchmark_4k_matrix/performance_matrix.json
outputs/benchmark_4k_matrix/performance_matrix.csv
outputs/benchmark_4k_matrix/performance_matrix.md
outputs/benchmark_4k_matrix_static_b4/performance_matrix.json
outputs/benchmark_4k_matrix_static_b4/performance_matrix.csv
outputs/benchmark_4k_matrix_static_b4/performance_matrix.md
```
