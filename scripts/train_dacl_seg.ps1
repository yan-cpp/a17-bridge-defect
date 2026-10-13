param(
    [string]$Model = "yolo11n-seg.pt",
    [string]$Data = "data/processed/dacl10k_subset/dataset.yaml",
    [int]$Epochs = 30,
    [int]$Imgsz = 640,
    [int]$Batch = 6,
    [string]$Name = "dacl_seg_baseline"
)
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$Python = if (Test-Path (Join-Path $ProjectRoot ".venv\Scripts\python.exe")) { Join-Path $ProjectRoot ".venv\Scripts\python.exe" } else { "python" }
Push-Location $ProjectRoot
try {
    & $Python -m a17.train.train_yolo --model $Model --data $Data --task segment --epochs $Epochs --imgsz $Imgsz --batch $Batch --device 0 --name $Name --project outputs/train
} finally {
    Pop-Location
}
