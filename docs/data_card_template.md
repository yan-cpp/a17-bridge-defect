# A17 数据卡

本文件由 `python -m a17.data.inspect` 生成。没有实际数据前，此文件记录模板和统计字段。

## 数据集概览

| 字段 | 值 |
|---|---|
| 数据集名称 | GYU-DET |
| 版本 | 待填写 |
| 下载时间 | 待填写 |
| 原始位置 | 待填写 |
| 许可证 | CC BY 4.0 |
| 使用角色 | 检测主数据 |

## 数据统计

| 指标 | train | validation | test |
|---|---|---|---|
| 图片数量 | - | - | - |
| 标签文件数量 | - | - | - |
| 病害实例数量 | - | - | - |

## 类别分布

| 类别 | 实例数量 |
|---|---|
| crack | - |
| spalling | - |
| exposed_rebar | - |
| efflorescence | - |
| honeycomb | - |

## 质量检查

- [ ] 图像和标签一一对应
- [ ] 类别索引在范围内
- [ ] 坐标在 0 到 1 之间
- [ ] 数据划分无同源泄漏
- [ ] 没有重复文件名

## 使用说明

- 训练集：`data/processed/images/train`
- 验证集：`data/processed/images/validation`
- 测试集：`data/processed/images/test`
- 检测标签：`data/processed/labels/{split}`
- 分割标签：`data/processed/masks/{split}`（有分割数据时）
