param(
    [string]$SourceModel = "weights/a17_seg_finetune_v5_last.pt",
    [int]$Imgsz = 640
)
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$Python = "python"
Push-Location $ProjectRoot
try {
    & $Python -m a17.freeze_v2 `
        --model $SourceModel `
        --calibration configs/calibration_v2.json `
        --quantification configs/quantification_v2.json `
        --output-model weights/a17_seg_v2_frozen.pt `
        --manifest configs/model_v2_freeze.json `
        --overwrite

    & $Python -m a17.deploy `
        --model weights/a17_seg_v2_frozen.pt `
        --format onnx `
        --imgsz $Imgsz `
        --half `
        --device 0

    & $Python -m a17.freeze_v2 `
        --model $SourceModel `
        --calibration configs/calibration_v2.json `
        --quantification configs/quantification_v2.json `
        --output-model weights/a17_seg_v2_frozen.pt `
        --manifest configs/model_v2_freeze.json `
        --onnx weights/a17_seg_v2_frozen.onnx `
        --overwrite
} finally {
    Pop-Location
}
