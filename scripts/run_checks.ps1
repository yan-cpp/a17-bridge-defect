param(
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if ([string]::IsNullOrWhiteSpace($Python)) {
    if (Test-Path $VenvPython) {
        $Python = $VenvPython
    } else {
        $Python = "python"
    }
}

$env:PYTHONPATH = Join-Path $ProjectRoot "src"
Push-Location $ProjectRoot
try {
    & $Python -m a17 validate-config
    & $Python -m a17 doctor
    & $Python -m a17 smoke
    & $Python -m unittest discover -s tests -v
} finally {
    Pop-Location
}
