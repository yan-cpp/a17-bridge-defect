# 补充数据集数据卡

## Concrete Crack Segmentation

- 原始文件：data/raw/Concrete Crack Segmentation.zip
- 处理结果：data/processed/concrete_crack_seg
- 许可证：CC BY 4.0
- 任务：二值裂缝语义分割
- 格式：Supervisely JSON，bitmap 为 zlib 压缩 PNG
- 转换后：train 400 / val 58，共 1010 个裂缝实例

## CrackSeg9k

- 原始文件：data/raw/CrackSeg9k.zip
- 处理结果：data/processed/crackseg9k
- 许可证：CC0
- 任务：二值裂缝分割（整合 10 个裂缝数据集）
- 采样规模：train 1651 / val 204，共 13507 个裂缝实例
- 说明：从 Vol1/Vol2 图像和 Vol1 的 Final_Masks/Masks 中按 basename 匹配，阈值 127 二值化。

## SDNET2018

- 原始文件：data/raw/senet2018.zip
- 处理结果：data/processed/sdnet2018_background
- 许可证：CC BY 4.0
- 任务：图像分类，无像素掩膜
- 用法：仅使用 Non-cracked 图像作为分割训练背景负样本。
- 采样规模：train 800 / val 100，空标签。

## 合并数据集

- 目录：data/processed/a17_combined
- train：7114 张（dacl10k 4263 + concrete 400 + crackseg9k 1651 + sdnet 800）
- val：979 张（dacl10k 617 + concrete 58 + crackseg9k 204 + sdnet 100）
- 标签类别：crack、spalling、exposed_rebar
