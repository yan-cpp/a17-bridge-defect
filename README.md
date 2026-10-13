# A17 Bridge Defect Detection

基于低空无人机视觉的桥梁表观病害精细化智能检测与分割项目，对应第十七届中国大学生服务外包创新创业大赛 A17 赛题。

第一阶段完成检测、实例分割、物理量化和本地 Demo 的完整闭环，目标类别为：

- `crack`：混凝土裂缝
- `spalling`：剥落或掉块
- `exposed_rebar`：钢筋裸露

第二阶段待补充类别：

- `rust`：钢结构锈蚀
- `efflorescence`：泛碱或渗水
- `honeycomb`：蜂窝面

## 当前进展

已完成：

- GYU-DET 完整检测模型（YOLO11n-detect，60 epochs）。
- dacl10k + CrackSeg9k + Concrete Crack Segmentation + SDNET2018 合并数据集的分割模型（YOLO11n-seg）。
- 高分辨率长训练和边界损失微调。
- mAP、Mask mAP、FPS 评估脚本和基线指标表。
- 实例 Mask 的物理量化（裂缝长度、P95/最大宽度、面积）。
- 本地 Demo：支持上传图片、示例图片点击、大图自动切块推理、标注图展示和 JSON 输出。
- 4K 滑动窗口推理与耗时基准、TensorRT/ONNX 导出脚本。

## 目录结构

```text
configs/       类别映射、标注规范、训练配置和指标模板
docs/          需求、环境、数据卡、指标和量化说明
src/a17/       Python 包（数据、训练、评估、量化、Demo）
tests/         单元测试
scripts/       数据转换、训练和评估的 PowerShell 脚本
data/          原始数据、中间数据和处理数据，不提交到 Git
outputs/       模型输出、评估结果和 Demo 产物，不提交到 Git
weights/       模型权重，不提交到 Git
runs/          训练运行产物，不提交到 Git
run_demo.bat   双击启动本地 Demo
```

## 环境与快速开始

```powershell
cd D:\zm\newA\a17-bridge-defect
$env:PYTHONPATH = "$PWD\src"
python -m a17 doctor
python -m a17 validate-config
python -m a17 smoke
python -m unittest discover -s tests -v
```

如需创建虚拟环境：

```powershell
.\scripts\bootstrap.ps1
```

## 数据

数据文件位于 `data/`，已通过 `.gitignore` 排除。主要数据集：

| 数据集 | 用途 | 处理结果 |
|---|---|---|
| GYU-DET | 检测主数据 | `data/processed/gyu_det` |
| dacl10k | 分割主数据 | `data/processed/dacl10k` |
| CrackSeg9k | 裂缝分割补充 | `data/processed/crackseg9k` |
| Concrete Crack Segmentation | 高分辨率裂缝补充 | `data/processed/concrete_crack_seg` |
| SDNET2018 | 背景负样本 | `data/processed/sdnet2018_background` |
| 合并数据集 | 分割训练 | `data/processed/a17_combined`（train 7114 / val 979） |

数据说明见：

- `docs/dataset_manifest.md`
- `docs/dacl10k_data_card.md`
- `docs/supplement_data_cards.md`
- `docs/gyu_det_data_card.md`
- `docs/annotation_spec.md`
- `configs/data_mapping.json`
- `configs/class_map.json`

## 训练

检测模型：

```powershell
python -m a17.train.train_yolo --task detect --model yolo11n.pt --data data/processed/gyu_det/dataset.yaml --epochs 60 --imgsz 640 --batch 8
```

分割模型：

```powershell
python -m a17.train.train_yolo --task segment --model yolo11n-seg.pt --data data/processed/a17_combined/dataset.yaml --epochs 40 --imgsz 1024 --batch 2
```

边界损失微调：

```powershell
python -m a17.train.train_boundary --model runs/segment/outputs/train/dacl_seg_highres_long/weights/best.pt --data data/processed/a17_combined/dataset.yaml --epochs 10 --imgsz 640 --batch 6 --boundary-weight 0.3
```

## 评估

检测评估：

```powershell
python -m a17.eval.evaluate --model runs/detect/outputs/train/gyu_det_full_v2/weights/best.pt --data data/processed/gyu_det/dataset.yaml --task detect --imgsz 640 --split val --name gyu_det_full_v2
```

分割评估：

```powershell
python -m a17.eval.evaluate --model runs/segment/outputs/train/dacl_seg_boundary_ft/weights/best.pt --data data/processed/a17_combined/dataset.yaml --task segment --imgsz 640 --split val --name dacl_seg_boundary_ft
```

4K 切块性能基准：

```powershell
python -m a17.eval.tile_inference --model runs/segment/outputs/train/dacl_seg_boundary_ft/weights/best.pt --image path/to/4k.jpg --tile 1024 --overlap 128 --imgsz 1024
```

TensorRT/ONNX 导出：

```powershell
python -m a17.deploy --model runs/segment/outputs/train/dacl_seg_boundary_ft/weights/best.pt --format engine --imgsz 1024 --half
```

## 量化

从实例 Mask 计算裂缝长度、P95 宽度、最大宽度和面积：

```powershell
python -m a17 quantify-mask --mask path/to/mask.png --scale 0.42
```

分割预测并导出 Mask、JSON 和量化结果：

```powershell
python -m a17 predict-seg --model runs/segment/outputs/train/dacl_seg_boundary_ft/weights/best.pt --source path/to/image.jpg --imgsz 1024 --scale 0.42
```

详见 `docs/quantification.md`。

## 本地 Demo

双击 `run_demo.bat`，或手动运行：

```powershell
cd D:\zm\newA\a17-bridge-defect
$env:PYTHONPATH = "$PWD\src"
python -m a17.demo_server --model runs/segment/outputs/train/dacl_seg_boundary_ft/weights/best.pt --port 8765
```

浏览器打开 `http://127.0.0.1:8765/`，可上传图片或点击示例图片，自动进行检测、分割、量化和可视化；大图自动滑动窗口切块。命令行窗口需要保持打开。

## 基线指标

检测（GYU-DET 验证集，YOLO11n-detect）：

- box mAP50：0.448
- box mAP50-95：0.257

分割（合并数据集验证集，YOLO11n-seg + 边界损失微调）：

- mask mAP50：0.175
- mask mAP50-95：0.064
- crack mask mAP50：0.082
- spalling mask mAP50：0.177
- exposed_rebar mask mAP50：0.266

完整指标见 `docs/baseline_metrics.md` 和 `outputs/eval/`。

## 注意事项

- 数据、权重和输出文件不进入 Git，Clone 后需重新下载数据并训练。
- dacl10k 使用 CC BY-NC 4.0 许可证，仅限非商业用途；正式参赛或商用前需确认合规。
- 4K 端到端 200ms 指标需在切块 + TensorRT 阶段单独验证。
