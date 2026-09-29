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
