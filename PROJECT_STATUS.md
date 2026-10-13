# A17 Project Status

## D1 Requirement And Scope Freeze

- [x] 第一版范围限定为 3 类病害。
- [x] 冻结类别：crack、spalling、exposed_rebar。
- [x] 冻结输入：无人机图像、图片目录或视频。
- [x] 冻结输出：类别、置信度、bbox、Mask、物理长度和最大宽度。
- [x] 定义 mAP、Mask IoU、速度和量化误差指标。
- [x] 记录数据源许可证风险。
- [x] 建立防数据泄漏规则。

## D2 Repository And Environment Baseline

- [x] 创建 Git 仓库目录结构。
- [x] 创建 VS Code settings、extensions、tasks 和 launch 配置。
- [x] 创建 Python 包和 CLI。
- [x] 创建配置文件和校验器。
- [x] 创建基础、开发和机器学习依赖清单。
- [x] 创建环境检查命令和冒烟测试。
- [x] 创建单元测试。

## Next D3

- [ ] 下载并盘点 GYU-DET、dacl10k、CrackSeg9k 等数据集。
- [ ] 核对许可证和下载完整性。
- [ ] 生成原始数据清单和数据统计表。
- [ ] 抽样检查图像质量、分辨率和重复样本。

## Week 2 Data Standard And Mapping

- [x] 标注规范：`docs/annotation_spec.md` 和 `configs/annotation_spec.yaml`。
- [x] 类别映射：`configs/data_mapping.json`。
- [x] 数据卡模板：`docs/data_card_template.md`。
- [x] GYU-DET 自动发现与统计脚本：`src/a17/data/inspect_gyu.py`。
- [x] GYU-DET 转 A17 YOLO 检测格式脚本：`src/a17/data/convert_gyu.py`。
- [x] 单元测试覆盖数据解析和 Mask IoU。

## Week 3 YOLO Baseline And Evaluation

- [x] 检测基线训练脚本：`src/a17/train/train_yolo.py`。
- [x] 分割基线配置：`configs/train_dacl_seg.yaml`。
- [x] mAP / Mask IoU / FPS 评测脚本：`src/a17/eval/evaluate.py` 和 `src/a17/eval/mask_iou.py`。
- [x] PowerShell 运行脚本：`scripts/train_gyu_det.ps1`、`scripts/evaluate_gyu_det.ps1`。

## Completed Weeks 2 And 3

- [x] 下载并解包 GYU-DET V4，完成类别映射和数据卡。
- [x] 安装 torch、torchvision、ultralytics、scikit-image、albumentations。
- [x] 建立 237 张训练 / 55 张验证 / 60 张测试的子集。
- [x] 跑通 YOLO11n 检测基线，5 epochs。
- [x] 生成 mAP / FPS 基线指标表。

## Completed Week 4 (dacl10k YOLO-seg baseline)

- [x] 解包并转换 dacl10k 为 A17 三类 YOLO-seg 数据集（train 4263 / val 617）。
- [x] 建立 dacl10k 分层快速子集（train 240 / val 40）。
- [x] 跑通 YOLO11n-seg 分割基线，30 epochs。
- [x] 生成 mAP / Mask mAP / FPS 基线指标表。
- [x] 更新评估脚本，支持 segment 任务的 mAP、Mask IoU 和 FPS。

## Completed Week 5 (supplement datasets + full dacl10k)

- [x] 转换 Concrete Crack Segmentation（train 400 / val 58，1010 实例）。
- [x] 转换 CrackSeg9k（train 1651 / val 204，13507 实例）。
- [x] 转换 SDNET2018 非裂缝背景负样本（train 800 / val 100）。
- [x] 合并完整 dacl10k 与三个补充数据集：a17_combined（train 7114 / val 979）。
- [x] 训练 YOLO11n-seg 合并完整数据基线，20 epochs。
- [x] 生成合并基线指标表，crack 类 Mask mAP50 提升至 0.072。

## Completed Week 6-7 (high-res longer training + quantification)

- [x] 高分辨率长训练完成：imgsz=1024，40 epochs，dacl_seg_highres_long。
- [x] 生成高分辨率基线指标表：mask mAP50 0.177。
- [x] 创建高分辨率训练配置和脚本。
- [x] 实现实例后处理与物理量化模块：src/a17/quantify.py。
- [x] 实现分割预测导出脚本：src/a17/eval/predict_seg.py。
- [x] 注册 CLI 命令 quantify-mask 和 predict-seg。

## In Progress Week 8 (boundary loss + full GYU detection)

- [x] 实现边界感知分割损失：src/a17/train/boundary_loss.py。
- [x] 边界损失冒烟训练通过。
- [x] 从高分辨率模型出发做边界损失微调（10 epochs），crack Mask mAP50 提升到 0.082。
- [x] 启动完整 GYU-DET 检测训练：gyu_det_full_v2，60 epochs。

## Completed Week 9-10 (full GYU detection + demo/tiling scaffolding)

- [x] GYU-DET 完整检测训练完成：gyu_det_full_v2，60 epochs。
- [x] GYU-DET 检测评估完成：box mAP50 0.448，mAP50-95 0.257。
- [x] 实现 4K 滑动窗口推理与耗时基准：src/a17/eval/tile_inference.py。
- [x] 实现 TensorRT/ONNX 导出脚本：src/a17/deploy.py。
- [x] 实现本地 Demo 服务器：src/a17/demo_server.py。

## Remaining

- [ ] 用边界模型做一次完整 4K 图片端到端性能验证（切块 + TensorRT）。
- [ ] 根据性能结果继续优化到 200ms 指标。
