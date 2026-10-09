param(
    [string]$Roots = "C:\Users\Lenovo\Downloads"
)
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$Python = if (Test-Path (Join-Path $ProjectRoot ".venv\Scripts\python.exe")) { Join-Path $ProjectRoot ".venv\Scripts\python.exe" } else { "python" }
Push-Location $ProjectRoot
try {
    & $Python -m a17.data.inspect_gyu --roots $Roots --output-json data/data_card.json --output-md data/data_card.md
} finally {
    Pop-Location
}
