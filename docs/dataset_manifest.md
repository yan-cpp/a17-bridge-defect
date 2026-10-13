# A17 数据集清单

## 原始数据

| 文件 | 来源数据集 | 用途 |
|---|---|---|
| data/raw/68827df9f367442c8be0c283e60ed3b7_V4.zip | GYU-DET | 桥梁表观病害检测主数据 |
| data/raw/dacl10k_v2_devphase.zip | dacl10k | 桥梁病害分割主数据 |
| data/raw/dacl10k_v2_testchallenge.zip | dacl10k | 未标注挑战集，暂不使用 |
| data/raw/dataverse_files.zip | CrackSeg9k | 裂缝分割预训练数据 |
| data/raw/archive.zip | SDNET2018 | 裂缝分类预训练数据 |
| data/raw/archive (2).zip | Concrete Crack Segmentation | 高分辨率裂缝分割数据 |

## 已处理数据

| 目录 | 说明 |
|---|---|
| data/processed/gyu_det | GYU-DET 转 A17 三类 YOLO 检测格式 |
| data/processed/gyu_det_subset | GYU-DET 快速子集 |
| data/processed/dacl10k | dacl10k 转 A17 三类 YOLO-seg 格式 |
| data/processed/dacl10k_subset | dacl10k 分层快速子集 |
| data/processed/concrete_crack_seg | Concrete Crack Segmentation 二值裂缝分割 |
| data/processed/crackseg9k | CrackSeg9k 裂缝分割采样 |
| data/processed/sdnet2018_background | SDNET2018 非裂缝背景负样本 |
| data/processed/a17_combined | 完整 dacl10k + 三个补充数据集的合并训练集 |

## 统一类别

| ID | Key | 中文 | 阶段 |
|---|---|---|---|
| 0 | crack | 混凝土裂缝 | 1 |
| 1 | spalling | 剥落或掉块 | 1 |
| 2 | exposed_rebar | 钢筋裸露 | 1 |
| 3 | rust | 钢结构锈蚀或混凝土腐蚀 | 2 |
| 4 | efflorescence | 泛碱或渗水 | 2 |
| 5 | honeycomb | 蜂窝面或孔洞 | 2 |

## 数据划分规则

- 按桥梁、拍摄批次或日期划分，避免同源泄漏。
- 第一阶段只使用 crack、spalling、exposed_rebar 三类。
- 训练/验证/测试默认比例 80% / 10% / 10%。
- 固定随机种子 42。
