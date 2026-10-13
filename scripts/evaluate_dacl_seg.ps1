param(
    [string]$Model = "runs/segment/outputs/train/dacl_seg_baseline/weights/best.pt",
    [string]$Data = "data/processed/dacl10k_subset/dataset.yaml",
    [int]$Imgsz = 640,
    [int]$Batch = 8,
    [string]$Name = "dacl_seg_baseline"
)
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$Python = if (Test-Path (Join-Path $ProjectRoot ".venv\Scripts\python.exe")) { Join-Path $ProjectRoot ".venv\Scripts\python.exe" } else { "python" }
Push-Location $ProjectRoot
try {
    & $Python -m a17.eval.evaluate --model $Model --data $Data --task segment --imgsz $Imgsz --batch $Batch --device 0 --split val --name $Name --output outputs/eval
} finally {
    Pop-Location
}
