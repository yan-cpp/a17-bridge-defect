# A17 V2 部署说明

## FP16 ONNX

已导出：

```text
weights/a17_seg_v2_frozen.onnx
```

属性：

- 输入：`images: float16 [1, 3, 640, 640]`
- 输出：`output0 [1, 39, 8400]`
- 输出：`output1 [1, 32, 160, 160]`
- 文件大小：约 5.6 MB

导出命令：

```powershell
cd D:\zm\newA\a17-bridge-defect
$env:PYTHONPATH = "$PWD\src"

python -m a17.deploy `
  --model weights/a17_seg_v2_frozen.pt `
  --format onnx `
  --imgsz 640 `
  --half `
  --device 0
```

## TensorRT Engine

TensorRT 10.13.3.9 已在 RTX 5060 Laptop GPU 上完成构建和推理验证。

生产路径使用静态 `imgsz=1024, batch=4, FP16`：

```powershell
python -m a17.deploy `
  --model weights/a17_seg_v2_frozen.pt `
  --format engine `
  --imgsz 1024 `
  --half `
  --batch 4 `
  --workspace 2 `
  --output weights/a17_seg_v2_frozen.engine `
  --device 0
```

需要同一个 engine 覆盖 batch 1–4 时，可追加 `--dynamic`。动态 engine 的构建和执行开销更高，当前测试中不优于静态 batch=4 engine。

安装依赖：

```powershell
python -m pip install -r requirements-tensorrt.txt
```

4K 性能与输出一致性结果见 `docs/performance_matrix_4k.md`。当前 RTX 5060 Laptop GPU 上的 4K 端到端 P50 约 1.45 秒，尚未达到 200 ms 验收目标。

## 运行时建议

- 4K 输入建议使用 `tile=1024`、`overlap=128`、`imgsz=1024`。
- GPU 批量切片优先使用 `batch_size=4`。
- 不建议使用多个 Python 线程同时调用同一个 CUDA 模型；请使用 batch 推理或独立进程。
- 实际长度计算必须读取 `configs/calibration_v2.json`；当前 `0.42 mm/px` 是临时标定值。
