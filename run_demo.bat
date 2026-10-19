@echo off
cd /d "D:\zm\newA\a17-bridge-defect"
set "PY=C:\Program Files\Python313\python.exe"
set "PYTHONPATH=%CD%\src"

"%PY%" -c "import cv2, ultralytics" 2>nul
if errorlevel 1 (
  echo.
  echo [Setup] Dependencies not found. Installing, first run can take several minutes...
  "%PY%" -m pip install -r requirements-ml.txt
  if errorlevel 1 (
    echo.
    echo [ERROR] Auto-install failed. Please run manually:
    echo   "%PY%" -m pip install -r requirements-ml.txt
    pause
    exit /b 1
  )
)

echo Starting A17 demo at http://127.0.0.1:8765
"%PY%" -m a17.demo_server --model weights/a17_seg_v2_frozen.pt --port 8765
pause
