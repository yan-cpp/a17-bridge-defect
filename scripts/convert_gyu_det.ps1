param(
    [string]$ArchivesRoot = "C:\Users\Lenovo\Downloads",
    [string]$OutputRoot = "data/processed/gyu_det"
)
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$env:YOLO_CONFIG_DIR = Join-Path $ProjectRoot ".ultralytics"
$Python = if (Test-Path (Join-Path $ProjectRoot ".venv\Scripts\python.exe")) { Join-Path $ProjectRoot ".venv\Scripts\python.exe" } else { "python" }
Push-Location $ProjectRoot
try {
    & $Python -m a17.data.convert_gyu --archives-root $ArchivesRoot --output-root $OutputRoot
} finally {
    Pop-Location
}
