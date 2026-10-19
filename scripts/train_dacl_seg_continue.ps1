param(
    [string]$Model = "runs/segment/outputs/train/dacl_seg_highres_long/weights/best.pt",
    [string]$Data = "data/processed/a17_combined/dataset.yaml",
    [int]$Epochs = 12,
    [int]$Imgsz = 1024,
    [int]$Batch = 2,
    [string]$Name = "dacl_seg_finetune_v5"
)
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$Python = if (Test-Path (Join-Path $ProjectRoot ".venv\Scripts\python.exe")) { Join-Path $ProjectRoot ".venv\Scripts\python.exe" } else { "python" }
Push-Location $ProjectRoot
try {
    & $Python -m a17.train.train_yolo `
        --model $Model `
        --data $Data `
        --task segment `
        --epochs $Epochs `
        --imgsz $Imgsz `
        --batch $Batch `
        --device 0 `
        --name $Name `
        --project outputs/train `
        --optimizer AdamW `
        --lr0 0.00005 `
        --lrf 0.1 `
        --cos-lr `
        --close-mosaic 0 `
        --mosaic 0.0 `
        --mixup 0.0 `
        --copy-paste 0.0 `
        --warmup-epochs 1 `
        --warmup-bias-lr 0.0 `
        --warmup-momentum 0.8 `
        --momentum 0.937 `
        --weight-decay 0.0005 `
        --workers 8 `
        --patience 6 `
        --save-period 2 `
        --exist-ok
} finally {
    Pop-Location
}
