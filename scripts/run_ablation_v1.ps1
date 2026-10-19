param(
    [string]$Data = "data/processed/dacl10k_subset/dataset.yaml",
    [int]$Epochs = 15,
    [int]$Imgsz = 512,
    [int]$Batch = 4
)
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$Python = "python"
Push-Location $ProjectRoot
try {
    $runs = @(
        @{
            Name = "ablation_v1_baseline"
            Model = "yolo11n-seg.pt"
            Pretrained = $null
            Boundary = 0.0
        },
        @{
            Name = "ablation_v1_p2"
            Model = "configs/models/a17_yolo11n_seg_p2.yaml"
            Pretrained = "yolo11n-seg.pt"
            Boundary = 0.0
        },
        @{
            Name = "ablation_v1_p2_ms"
            Model = "configs/models/a17_yolo11n_seg_p2_ms.yaml"
            Pretrained = "yolo11n-seg.pt"
            Boundary = 0.0
        },
        @{
            Name = "ablation_v1_p2_ms_boundary"
            Model = "configs/models/a17_yolo11n_seg_p2_ms.yaml"
            Pretrained = "yolo11n-seg.pt"
            Boundary = 0.3
        }
    )

    foreach ($run in $runs) {
        $trainArgs = @(
            "-B", "-m", "a17.train.train_yolo",
            "--task", "segment",
            "--model", $run.Model,
            "--data", $Data,
            "--epochs", $Epochs,
            "--imgsz", $Imgsz,
            "--batch", $Batch,
            "--device", "0",
            "--project", "outputs/train",
            "--name", $run.Name,
            "--patience", "5",
            "--save-period", "5",
            "--optimizer", "AdamW",
            "--lr0", "0.001",
            "--lrf", "0.01",
            "--close-mosaic", "5",
            "--mosaic", "0.5",
            "--warmup-epochs", "1",
            "--warmup-bias-lr", "0.0",
            "--workers", "4",
            "--exist-ok"
        )
        if ($run.Pretrained) {
            $trainArgs += @("--pretrained", $run.Pretrained)
        }
        if ($run.Boundary -gt 0) {
            $trainArgs += @("--boundary-weight", $run.Boundary)
        }
        & $Python @trainArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Ablation training failed: $($run.Name)"
        }

        $evalArgs = @(
            "-B", "-m", "a17.eval.evaluate",
            "--model", "runs/segment/outputs/train/$($run.Name)/weights/best.pt",
            "--data", $Data,
            "--task", "segment",
            "--imgsz", $Imgsz,
            "--batch", $Batch,
            "--device", "0",
            "--split", "val",
            "--name", $run.Name,
            "--output", "outputs/ablation_v1"
        )
        & $Python @evalArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Ablation evaluation failed: $($run.Name)"
        }
    }
} finally {
    Pop-Location
}
