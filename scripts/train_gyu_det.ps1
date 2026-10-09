param(
    [string]$Model = "yolo11n.pt",
    [string]$Data = "data/processed/gyu_det/dataset.yaml",
    [int]$Epochs = 50,
    [int]$Imgsz = 1024,
    [int]$Batch = 2
)
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$Python = if (Test-Path (Join-Path $ProjectRoot ".venv\Scripts\python.exe")) { Join-Path $ProjectRoot ".venv\Scripts\python.exe" } else { "python" }
Push-Location $ProjectRoot
try {
    & $Python -m a17.train.train_yolo --model $Model --data $Data --task detect --epochs $Epochs --imgsz $Imgsz --batch $Batch --name gyu_det_baseline
} finally {
    Pop-Location
}
