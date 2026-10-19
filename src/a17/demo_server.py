"""Local HTTP demo for A17 segmentation and quantification.

Convenience features:
- upload an image directly from the browser (no need to type a path)
- clickable sample images from the processed datasets
- automatic tiling for large images so thin cracks are not lost
Localhost only.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs, quote

import cv2
import numpy as np

SAMPLE_DIRS = [
    "data/processed/dacl10k_subset/images/val",
    "data/processed/concrete_crack_seg/images/train",
    "data/processed/crackseg9k/images/val",
    "data/processed/sdnet2018_background/images/val",
]

COLORS = {
    "crack": (0, 0, 255),
    "spalling": (255, 0, 0),
    "exposed_rebar": (0, 255, 0),
}
DEFAULT_COLOR = (255, 255, 0)


def sample_links() -> str:
    items = []
    for directory in SAMPLE_DIRS:
        base = Path(directory)
        if not base.exists():
            continue
        files = [p for p in base.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")][:4]
        for file_path in files:
            href = "/predict?image=" + quote(str(file_path.resolve()).replace("\\", "/"), safe="/:") + "&scale=0.42"
            items.append(f'<a href="{href}">{base.parent.name}/{base.name}/{file_path.name}</a>')
    if not items:
        return "<p>暂无示例图片</p>"
    return "<div class='links'>" + "<br>".join(items) + "</div>"


HTML = """<!doctype html>
<meta charset="utf-8">
<title>A17 Bridge Defect Demo</title>
<style>
  body{font-family:Arial,sans-serif;max-width:980px;margin:28px auto;color:#1f2937}
  h2{margin-bottom:6px}
  .card{background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:18px;margin-top:16px}
  input[type=file]{padding:6px}
  button{background:#1d4ed8;color:#fff;border:0;border-radius:6px;padding:8px 16px;cursor:pointer}
  a{color:#1d4ed8;text-decoration:none}
  .links a{display:inline-block;margin:4px 8px 4px 0;padding:4px 8px;border:1px solid #dbeafe;background:#eff6ff;border-radius:6px;font-size:13px}
  img{max-width:100%;border:1px solid #e2e8f0;border-radius:8px;margin-top:12px}
  ul{margin-top:10px}
</style>
<h2>A17 桥梁病害分割 Demo</h2>
<p>上传图片或点击示例图片即可运行，无需输入路径。</p>

<div class="card">
  <form action="/upload" method="post" enctype="multipart/form-data">
    <div>选择图片：<input type="file" name="file" accept="image/*"></div>
    <div style="margin-top:8px">毫米每像素：<input name="scale" type="number" step="0.01" value="0.42"></div>
    <div style="margin-top:8px">置信度阈值：<input name="conf" type="number" min="0.01" max="0.95" step="0.01" value="{conf_default}"></div>
    <div style="margin-top:4px;color:#64748b;font-size:13px">阈值越低越容易检出弱裂缝，但误检也会增加。建议先用 0.10 检查，再按验证结果调整。</div>
    <div style="margin-top:10px"><button type="submit">上传并运行</button></div>
  </form>
</div>

<div class="card">
  <div>示例图片：</div>
  {samples}
</div>
"""


def draw_boxes(image: np.ndarray, records: list[dict]) -> np.ndarray:
    canvas = image.copy()
    for item in records:
        color = COLORS.get(item.get("class_key"), DEFAULT_COLOR)
        bbox = item.get("bbox_xyxy")
        if not bbox:
            continue
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{item.get('class_key')} {item.get('confidence', 0):.2f}"
        cv2.putText(canvas, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return canvas


def draw_predictions(image: np.ndarray, records: list[dict]) -> np.ndarray:
    """Draw stitched masks and boxes on the original image."""
    overlay = image.copy()
    for item in records:
        color = COLORS.get(item.get("class_key"), DEFAULT_COLOR)
        mask = item.get("mask")
        if mask is None and item.get("mask_path"):
            mask_path = Path(item["mask_path"])
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            overlay[np.asarray(mask) > 0] = color
        bbox = item.get("bbox_xyxy")
        if bbox:
            x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            label = f"{item.get('class_key')} {item.get('confidence', 0):.2f}"
            cv2.putText(
                overlay,
                label,
                (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
    return cv2.addWeighted(image, 0.45, overlay, 0.55, 0)


def draw_annotations(image_path: Path, predictions: list[dict]) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image for overlay: {image_path}")
    overlay = image.copy()
    for item in predictions:
        color = COLORS.get(item.get("class_key"), DEFAULT_COLOR)
        mask_path = item.get("mask_path")
        if mask_path and Path(mask_path).exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask = (mask > 127).astype(np.uint8)
                overlay[mask > 0] = color
        bbox = item.get("bbox_xyxy")
        if bbox:
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            label = f"{item.get('class_key')} {item.get('confidence', 0):.2f}"
            cv2.putText(overlay, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return cv2.addWeighted(image, 0.45, overlay, 0.55, 0)


def encode_jpeg_data_uri(image: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise ValueError("Failed to encode annotated image.")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def result_html(data_uri: str, records: list[dict], tiled: bool, confidence: float) -> str:
    rows = []
    for item in records:
        text = f"{item.get('class_key')} conf={item.get('confidence')} "
        if "length_mm" in item:
            text += f"L={item.get('length_mm')}mm W={item.get('width_mm_p95')}mm"
        if int(item.get("bridge_count", 0) or 0) > 0:
            text += f"（拼接 {item['bridge_count']} 处间隙）"
        rows.append(f"<li>{text}</li>")
    if not rows:
        rows.append("<li>未检出目标。可尝试把置信度阈值暂时降到 0.08 或更低，并人工复核。</li>")
    low_confidence_count = sum(float(item.get("confidence", 0)) < 0.2 for item in records)
    warning = ""
    if low_confidence_count:
        warning = (
            f"<p style='color:#b45309'><strong>提示：</strong>"
            f"有 {low_confidence_count} 个低置信度结果（低于 0.20），必须人工复核。</p>"
        )
    mode = "大图滑动窗口切块推理 + Mask 拼接" if tiled else "整图推理（框 + Mask + 量化）"
    note = f"<p>模式：{mode}；置信度阈值：{confidence:.2f}</p>"
    return f"""<!doctype html>
<meta charset="utf-8">
<title>A17 Result</title>
<style>body{{font-family:Arial,sans-serif;max-width:980px;margin:28px auto}}img{{max-width:100%;border:1px solid #e2e8f0;border-radius:8px}}</style>
<h2>检测结果</h2>
{note}
{warning}
<img src="{data_uri}" alt="annotated">
<ul>{''.join(rows)}</ul>
<a href="/">返回</a>
"""


def parse_multipart(body: bytes, boundary: str) -> dict:
    marker = b"--" + boundary.encode("latin-1")
    fields: dict[str, bytes] = {}
    for part in body.split(marker):
        if b"\r\n\r\n" not in part:
            continue
        header, _, data = part.partition(b"\r\n\r\n")
        data = data.rstrip(b"\r\n")
        header_text = header.decode("latin-1", "ignore")
        name_match = re.search(r'name="([^"]+)"', header_text)
        filename_match = re.search(r'filename="([^"]+)"', header_text)
        if filename_match:
            fields["file"] = data
            fields["filename"] = filename_match.group(1).encode("latin-1")
        elif name_match:
            fields[name_match.group(1)] = data.split(b"\r\n")[0]
    return fields


def build_handler(model_path: Path, scale_default: float, conf_default: float = 0.10):
    from ultralytics import YOLO
    model = YOLO(str(model_path))
    from a17.eval.predict_seg import run
    from a17.eval.tile_inference import predict_tiled, public_records

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                page = HTML.replace("{samples}", sample_links())
                page = page.replace("{conf_default}", f"{conf_default:.2f}")
                self._send_html(page)
            elif parsed.path == "/predict":
                self._handle_predict(parsed)
            else:
                self._send_html("not found", status=404)

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path != "/upload":
                self._send_html("not found", status=404)
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            content_type = self.headers.get("Content-Type", "")
            boundary_match = re.search(r"boundary=([^;]+)", content_type)
            if not boundary_match:
                self._send_json({"ok": False, "error": "缺少 multipart boundary。"}, status=400)
                return
            fields = parse_multipart(body, boundary_match.group(1).strip('"'))
            file_bytes = fields.get("file")
            if not file_bytes:
                self._send_json({"ok": False, "error": "未收到文件，请选择一张图片。"}, status=400)
                return
            filename = fields.get("filename", b"upload.jpg").decode("latin-1", "ignore")
            try:
                scale = float(
                    (fields.get("scale") or str(scale_default).encode()).decode(
                        "utf-8",
                        "ignore",
                    )
                    or scale_default
                )
                confidence = float(
                    (fields.get("conf") or str(conf_default).encode()).decode(
                        "utf-8",
                        "ignore",
                    )
                    or conf_default
                )
            except ValueError:
                self._send_json({"ok": False, "error": "毫米每像素或置信度阈值格式不正确。"}, status=400)
                return
            if not 0.01 <= confidence <= 0.95:
                self._send_json({"ok": False, "error": "置信度阈值必须在 0.01 到 0.95 之间。"}, status=400)
                return
            upload_dir = Path("outputs/demo/_uploads")
            upload_dir.mkdir(parents=True, exist_ok=True)
            image_path = upload_dir / filename
            image_path.write_bytes(file_bytes)
            self._serve_prediction(image_path, scale, confidence, False)

        def _handle_predict(self, parsed):
            query = parse_qs(parsed.query)
            image = (query.get("image", [""])[0] or "").strip().strip('"')
            want_json = query.get("format", [""])[0] == "json"
            scale = float(query.get("scale", [str(scale_default)])[0] or scale_default)
            try:
                confidence = float(query.get("conf", [str(conf_default)])[0] or conf_default)
            except ValueError:
                self._send_json({"ok": False, "error": "置信度阈值格式不正确。"}, status=400)
                return
            if not 0.01 <= confidence <= 0.95:
                self._send_json({"ok": False, "error": "置信度阈值必须在 0.01 到 0.95 之间。"}, status=400)
                return
            if not image:
                self._send_json({"ok": False, "error": "image 参数为空，请上传图片或点击示例图片。"}, status=400)
                return
            image_path = Path(image)
            if not image_path.is_file():
                self._send_json({"ok": False, "error": f"找不到图片文件：{image}"}, status=404)
                return
            self._serve_prediction(image_path, scale, confidence, want_json)

        def _serve_prediction(
            self,
            image_path: Path,
            scale: float,
            confidence: float,
            want_json: bool,
        ):
            image = cv2.imread(str(image_path))
            if image is None:
                self._send_json({"ok": False, "error": f"无法读取图片：{image_path}"}, status=400)
                return

            height, width = image.shape[:2]
            use_tile = max(height, width) > 1024

            if use_tile:
                out_dir = Path("outputs/demo") / image_path.stem
                try:
                    records = predict_tiled(
                        model,
                        image,
                        tile=1024,
                        overlap=128,
                        imgsz=1024,
                        conf=confidence,
                        include_masks=True,
                        merge_bridge_px=24,
                        merge_bridge_min_confidence=0.10,
                        masks_dir=out_dir / "masks",
                        image_name=image_path.name,
                        scale=scale,
                        device="0",
                        half=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    self._send_json({"ok": False, "error": str(exc)}, status=500)
                    return
                if want_json:
                    self._send_json(
                        {
                            "ok": True,
                            "image": str(image_path),
                            "scale": scale,
                            "confidence_threshold": confidence,
                            "tiled": True,
                            "predictions": public_records(records),
                        }
                    )
                    return
                try:
                    annotated = draw_predictions(image, records)
                    data_uri = encode_jpeg_data_uri(annotated)
                except Exception as exc:  # noqa: BLE001
                    self._send_json({"ok": False, "error": f"渲染标注图失败: {exc}"}, status=500)
                    return
                self._send_html(
                    result_html(
                        data_uri,
                        records,
                        tiled=True,
                        confidence=confidence,
                    )
                )
                return

            out_dir = Path("outputs/demo") / image_path.stem
            try:
                run(
                    model_path,
                    image_path,
                    out_dir,
                    imgsz=1024,
                    conf=confidence,
                    iou=0.5,
                    device="0",
                    scale=scale,
                    half=True,
                )
                payload = json.loads((out_dir / "predictions.json").read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": str(exc)}, status=500)
                return

            if want_json:
                self._send_json(
                    {
                        "ok": True,
                        "image": str(image_path),
                        "scale": scale,
                        "confidence_threshold": confidence,
                        "tiled": False,
                        "predictions": payload,
                    }
                )
                return

            try:
                annotated = draw_annotations(image_path, payload)
                data_uri = encode_jpeg_data_uri(annotated)
            except Exception as exc:  # noqa: BLE001
                self._send_json({"ok": False, "error": f"渲染标注图失败: {exc}"}, status=500)
                return
            self._send_html(
                result_html(
                    data_uri,
                    payload,
                    tiled=False,
                    confidence=confidence,
                )
            )

        def _send_html(self, body, status=200):
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_json(self, obj, status=200):
            data = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    return Handler


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the local A17 demo server.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--scale", type=float, default=0.42)
    parser.add_argument("--conf", type=float, default=0.10)
    args = parser.parse_args(argv)

    handler = build_handler(Path(args.model), args.scale, args.conf)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"Demo server at http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
