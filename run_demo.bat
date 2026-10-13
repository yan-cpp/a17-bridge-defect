@echo off
cd /d "D:\zm\newA\a17-bridge-defect"
set PYTHONPATH=%CD%\src
echo Starting A17 demo at http://127.0.0.1:8765
python -m a17.demo_server --model runs/segment/outputs/train/dacl_seg_boundary_ft/weights/best.pt --port 8765
pause
