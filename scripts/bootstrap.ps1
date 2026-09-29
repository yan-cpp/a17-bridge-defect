param(
    [string]$Python = "python",
    [switch]$InstallMachineLearning
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvPath = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"

if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
    throw "Python command not found: $Python"
}

if (-not (Test-Path $VenvPython)) {
    & $Python -m venv $VenvPath
}

& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements-dev.txt")

if ($InstallMachineLearning) {
    & $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements-ml.txt")
}

$env:PYTHONPATH = Join-Path $ProjectRoot "src"
& $VenvPython -m a17 validate-config
& $VenvPython -m a17 doctor
& $VenvPython -m unittest discover -s (Join-Path $ProjectRoot "tests") -v
