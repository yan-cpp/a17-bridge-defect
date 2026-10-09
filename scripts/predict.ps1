param(
    [Parameter(Mandatory=$true)][string]$Source,
    [string]$Model = "runs/detect/outputs/train/gyu_det_full/weights/best.pt",
    [string]$Output = "outputs/predict",
    [int]$Imgsz = 640,
    [double]$Conf = 0.25
)
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$env:YOLO_CONFIG_DIR = Join-Path $ProjectRoot ".ultralytics"
$Python = "python"
Push-Location $ProjectRoot
try {
    & $Python -m a17.eval.predict --model $Model --source $Source --output $Output --imgsz $Imgsz --conf $Conf
} finally {
    Pop-Location
}
