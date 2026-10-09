param(
    [string]$Model = "outputs/train/gyu_det_baseline/weights/best.pt",
    [string]$Data = "data/processed/gyu_det/dataset.yaml",
    [int]$Imgsz = 1024
)
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$env:YOLO_CONFIG_DIR = Join-Path $ProjectRoot ".ultralytics"
$Python = if (Test-Path (Join-Path $ProjectRoot ".venv\Scripts\python.exe")) { Join-Path $ProjectRoot ".venv\Scripts\python.exe" } else { "python" }
Push-Location $ProjectRoot
try {
    & $Python -m a17.eval.evaluate --model $Model --data $Data --task detect --imgsz $Imgsz --split test --name gyu_det_baseline --output outputs/eval
} finally {
    Pop-Location
}
