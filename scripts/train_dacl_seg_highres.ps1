param(
    [string]$Model = "yolo11n-seg.pt",
    [string]$Data = "data/processed/a17_combined/dataset.yaml",
    [int]$Epochs = 40,
    [int]$Imgsz = 1024,
    [int]$Batch = 2,
    [string]$Name = "dacl_seg_highres_long"
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
