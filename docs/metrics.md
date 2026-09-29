# Output Schema

## Top-Level Structure

```json
{
  "schema_version": "0.1.0",
  "image_id": "bridge_001.jpg",
  "image_width": 3840,
  "image_height": 2160,
  "model_version": "baseline-v0",
  "config_version": "0.1.0",
  "timing_ms": {
    "preprocess_ms": 0.0,
    "inference_ms": 0.0,
    "postprocess_ms": 0.0,
    "quantification_ms": 0.0,
    "total_ms": 0.0
  },
  "defects": [
    {
      "instance_id": 1,
      "class_key": "crack",
      "class_name_zh": "混凝土裂缝",
      "confidence": 0.93,
      "bbox_xyxy": [120, 340, 580, 410],
      "mask_path": "masks/bridge_001_001.png",
      "length_mm": 182.4,
      "width_mm": 1.7,
      "width_statistic": "p95",
      "scale_mm_per_pixel": 0.42,
      "quantification_valid": true
    }
  ]
}
```

## Field Rules

| Field | Type | Required | Rule |
|---|---|---|---|
| schema_version | string | yes | 当前为0.1.0 |
| image_id | string | yes | 原图唯一标识 |
| image_width | integer | yes | 原图像素宽度 |
| image_height | integer | yes | 原图像素高度 |
| model_version | string | yes | 模型或权重版本 |
| config_version | string | yes | 配置版本 |
| timing_ms.total_ms | number | yes | 端到端耗时 |
| defects | array | yes | 无病害时为空数组 |
| class_key | string | yes | 必须来自class_map.json |
| confidence | number | yes | 范围0至1 |
| bbox_xyxy | number[4] | yes | 原图左上和右下坐标 |
| mask_path | string | yes | Mask文件位置，PNG二值图 |
| length_mm | number or null | yes | 无法量化时为null |
| width_mm | number or null | yes | 使用稳健统计值 |
| scale_mm_per_pixel | number or null | yes | 来源必须可追溯 |
| quantification_valid | boolean | yes | 标定或Mask不合格时返回false |

## Versioning

- 类别新增或重命名：递增主版本。
- 字段含义变化：递增主版本。
- 新增可选字段：递增次版本。
- 文档和注释修正：递增补丁版本。
