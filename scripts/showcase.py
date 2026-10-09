import json
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(r"D:\zm\newA\a17-bridge-defect")
MODEL = ROOT / "runs/detect/outputs/train/gyu_det_full/weights/best.pt"
DATA = ROOT / "data/processed/gyu_det/dataset.yaml"
TEST_IMAGES = ROOT / "data/processed/gyu_det/images/test"
OUT = ROOT / "outputs/showcase"
OUT.mkdir(parents=True, exist_ok=True)

images = sorted(TEST_IMAGES.glob("*.jpg"))[:6] + sorted(TEST_IMAGES.glob("*.JPG"))[:2]
images = sorted(set(images))[:6]

model = YOLO(str(MODEL))
results = model.predict(
    source=[str(p) for p in images],
    imgsz=640,
    conf=0.25,
    iou=0.5,
    device=0,
    save=True,
    project=str(ROOT / "outputs/showcase"),
    name="run",
    exist_ok=True,
)

summary = []
for image, result in zip(images, results):
    names = result.names
    counts = {}
    for cls in result.boxes.cls.tolist():
        name = names[int(cls)]
        counts[name] = counts.get(name, 0) + 1
    summary.append({"image": image.name, "detections": counts})

(OUT / "showcase_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
