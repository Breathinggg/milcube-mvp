@echo off
echo ========================================
echo MilCube-MVP Re-ID Setup
echo ========================================
echo.

set "PY=D:\Python3.11\python.exe"

if not exist "%PY%" (
  echo [ERROR] Python not found: %PY%
  echo Please edit this script to set the correct Python path.
  pause
  exit /b 1
)

echo [1/2] Checking dependencies...
"%PY%" -c "import onnxruntime" 2>nul
if errorlevel 1 (
  echo Installing onnxruntime-gpu...
  "%PY%" -m pip install onnxruntime-gpu
)

echo.
echo [2/2] Setting up Re-ID model...
"%PY%" scripts\download_reid_model.py

echo.
echo ========================================
echo Setup complete!
echo ========================================
echo.
echo To run MilCube with Re-ID:
echo   run_all.bat
echo.
echo To run without Re-ID (color histogram):
echo   Add --no_reid flag to main.py
echo.
pause
