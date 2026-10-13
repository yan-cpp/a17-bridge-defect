# dacl10k 数据卡

## 基本信息

| 项目 | 内容 |
|---|---|
| 数据集 | dacl10k v2 devphase |
| 发布方 | Flotzinger, Rösch, Braml (WACV 2024) |
| 原始文件 | data/raw/dacl10k_v2_devphase.zip |
| 处理结果 | data/processed/dacl10k |
| 许可证 | CC BY-NC 4.0（非商业） |
| 任务 | 桥梁病害多标签语义分割 |
| 原始类别数 | 19（13类病害 + 6类构件） |

## A17 第一阶段类别映射

| dacl10k 原始标签 | A17 内部类别 | 是否保留 |
|---|---|---|
| Crack | crack | 是 |
| ACrack | crack | 是 |
| Spalling | spalling | 是 |
| ExposedRebars | exposed_rebar | 是 |
| Rust | rust | 否（第二阶段） |
| WConccor | rust | 否（第二阶段） |
| Efflorescence | efflorescence | 否（第二阶段） |
| Wetspot | efflorescence | 否（第二阶段） |
| Rockpocket / Cavity / Hollowareas | honeycomb | 否（第二阶段） |
| Bearing / EJoint / Drainage 等 | 构件类 | 否 |

## 转换后统计

| Split | 图片数 | 实例数 | 标签文件数 |
|---|---|---|---|
| train | 4263 | 13767 | 4263 |
| val | 617 | 2167 | 617 |

类别实例分布：

| 类别 | 实例数 |
|---|---|
| crack | 4050 |
| spalling | 9884 |
| exposed_rebar | 2000 |

## 处理说明

- 读取 dacl10k 的 labelme 风格 JSON 多边形标注。
- 每个保留的多边形写为一条 YOLO-seg 实例标签：`class x1 y1 x2 y2 ...`。
- 坐标按 imageWidth / imageHeight 归一化到 0-1。
- 只保留 A17 第一阶段三类；其他病害和构件类丢弃。
- 没有第一阶段病害实例的图像不写入训练集。
- 原 dacl10k 是语义分割，多实例掩膜是通过“一个多边形一个实例”方式从标注中提取的。

## 子集

为快速基线训练，建立了分层子集：

- data/processed/dacl10k_subset
- train：240 张 / 908 实例
- val：40 张 / 188 实例
- 生成脚本：src/a17/data/make_subset.py
- 随机种子：42
