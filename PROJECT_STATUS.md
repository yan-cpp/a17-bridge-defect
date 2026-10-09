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

## Blocked On External Inputs

- [ ] 找到 GYU-DET 的 `train.zip`、`valid.zip`、`test.zip` 实际路径。
- [ ] 安装 `torch`、`torchvision`、`ultralytics` 后执行真实基线训练。
