# A17 Bridge Defect Detection

A17 赛题工程项目骨架，用于实现基于低空无人机视觉的桥梁表观病害精细化智能检测算法。

## 第一阶段范围

第一阶段仅冻结以下三类病害，先完成检测、实例分割、物理量化和演示闭环：

- `crack`：混凝土裂缝
- `spalling`：剥落或掉块
- `exposed_rebar`：钢筋裸露

以下类别保留到第二阶段：

- `rust`：钢结构锈蚀
- `efflorescence`：泛碱或渗水
- `honeycomb`：蜂窝面

## 当前阶段

- D1：需求、类别、输出协议、指标和许可证规则已落盘。
- D2：Git/VSCode 工程骨架、目录、依赖和运行环境检查已实现。

## 快速开始

```powershell
cd D:\zm\newA\a17-bridge-defect
$env:PYTHONPATH = "$PWD\src"
python -m a17 doctor
python -m a17 validate-config
python -m a17 smoke
python -m unittest discover -s tests -v
```

如果尚未创建虚拟环境：

```powershell
.\scripts\bootstrap.ps1
```

## 目录

```text
.vscode/       VS Code 设置、任务和调试配置
configs/       类别映射、指标、运行时和数据模板
docs/          D1/D2 需求与环境文档
src/a17/       项目 Python 包
tests/         单元测试
scripts/       环境初始化和检查脚本
data/          原始数据和中间数据，不提交到 Git
outputs/       模型输出、报告和量化结果，不提交到 Git
weights/       模型权重，不提交到 Git
```

## 当前验收

- 配置文件可校验。
- `python -m a17 doctor` 可检查 Python、GPU 和可选依赖。
- `python -m a17 smoke` 可生成一条符合输出协议的结果。
- 单元测试可独立运行。
