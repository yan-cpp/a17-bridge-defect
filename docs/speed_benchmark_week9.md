# 第 9 周 4K 速度基准报告

## 环境

- GPU：NVIDIA GeForce RTX 5060 Laptop GPU，8151 MiB
- PyTorch：2.11.0+cu128
- 输入：`4128 x 3096`
- 切片：`tile=1024`、`overlap=128`、`imgsz=1024`
- tiles：20

## PyTorch 切片结果

| 精度 | batch | workers | 平均 ms | P50 ms | P95 ms | 全图 FPS |
|---|---:|---:|---:|---:|---:|---:|
| FP16 | 4 | 1 | 771.77 | 760.66 | 794.99 | 1.296 |
| FP32 | 4 | 1 | 825.53 | 829.29 | 832.13 | 1.211 |

FP16 相对 FP32 的端到端提升约：

```text
(825.53 - 771.77) / 825.53 = 6.51%
```

## 并行策略

- 批量切片 `batch_size=4` 是当前设备最优路径。
- `workers=2/4` 使用多个 Python 线程调用同一个 CUDA 模型会因 GPU 竞争变慢。
- 推荐生产路径：单进程 + batch 切块 + FP16。

## ONNX FP16 检查

- 文件：`weights/a17_seg_v2_frozen.onnx`
- 输入类型：`float16`
- 输出：`output0 [1,39,8400]`、`output1 [1,32,160,160]`
- ONNX Runtime CPU 回退：`17.374 ms / 57.56 FPS @ 640`

当前机器 ONNX Runtime CUDA Provider 缺少对应 cuDNN/CUDA 运行库，因此没有使用 ONNX Runtime GPU 数据替代 PyTorch CUDA 结果。

## 原始结果

- `outputs/benchmark_4k_final/speed_benchmark.json`
- `outputs/benchmark_4k_fp32_final/speed_benchmark.json`
- `outputs/benchmark_onnx/benchmark.json`
