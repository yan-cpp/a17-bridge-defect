$ErrorActionPreference = "Stop"

$env:PYTHONPATH = Join-Path $PSScriptRoot "..\src"
$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $projectRoot

$python = if ($env:A17_PYTHON) { $env:A17_PYTHON } else { "python" }
$model = "weights\a17_seg_v2_frozen.pt"
$engine = "weights\a17_seg_v2_frozen_dynamic_b4.engine"
$image = "data\processed\a17_combined\images\val\dacl10k_v2_validation_0545.jpg"

& $python -m a17.deploy `
  --model $model `
  --format engine `
  --imgsz 1024 `
  --half `
  --dynamic `
  --batch 4 `
  --workspace 2 `
  --output $engine `
  --device 0

& $python -m a17.eval.benchmark_matrix `
  --pytorch-model $model `
  --engine-model $engine `
  --image $image `
  --tile 1024 `
  --overlap 128 `
  --imgsz 1024 `
  --batch-sizes 1,2,4 `
  --repeat 5 `
  --device 0 `
  --output outputs\benchmark_4k_matrix
