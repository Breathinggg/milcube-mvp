@echo off
setlocal enabledelayedexpansion

REM ======== MilCube MVP - Fall Detection Dataset ========
set "ROOT=%~dp0"
set "PY=D:\Python3.11\python.exe"

set "CUDNN_BIN=C:\Program Files\NVIDIA\CUDNN\v9.17\bin\12.9"

set "CAM5=fall1.mp4"
set "CAM6=fall2.mp4"
set "MODEL=models\yolov8n.onnx"

REM ports
set "PORT5=5000"
set "PORT6=5001"
set "DASH=9000"

REM ======== env: make sure cuDNN 12.9 wins DLL search ========
set "PATH=%CUDNN_BIN%;%PATH%"

echo [MilCube-MVP] Fall Detection Dataset Mode
echo [MilCube-MVP] ROOT = %ROOT%
echo [MilCube-MVP] PY   = %PY%
echo [MilCube-MVP] cuDNN= %CUDNN_BIN%
echo.

REM ======== sanity check ========
if not exist "%PY%" (
  echo [ERROR] Python not found: %PY%
  pause
  exit /b 1
)
if not exist "%ROOT%\%MODEL%" (
  echo [ERROR] Model not found: %ROOT%\%MODEL%
  pause
  exit /b 1
)
if not exist "%ROOT%\%CAM5%" (
  echo [WARN] fall1 not found: %ROOT%\%CAM5%
)
if not exist "%ROOT%\%CAM6%" (
  echo [WARN] fall2 not found: %ROOT%\%CAM6%
)

REM ======== start processes in separate consoles ========
cd /d "%ROOT%"

echo [MilCube-MVP] Starting fall1 on :%PORT5% ...
start "MilCube-fall1" cmd /k ""%PY%" main.py --src "%CAM5%" --model "%MODEL%" --gpu --min_infer_ms 0 --port %PORT5% --name fall1 --headless"

echo [MilCube-MVP] Starting fall2 on :%PORT6% ...
start "MilCube-fall2" cmd /k ""%PY%" main.py --src "%CAM6%" --model "%MODEL%" --gpu --min_infer_ms 0 --port %PORT6% --name fall2 --headless"

echo [MilCube-MVP] Starting dashboard on :%DASH% ...
start "MilCube-dashboard" cmd /k ""%PY%" dashboard_server.py"

echo.
echo [MilCube-MVP] OPEN:
echo   fall1 video: http://127.0.0.1:%PORT5%/video
echo   fall2 video: http://127.0.0.1:%PORT6%/video
echo   dashboard  : http://127.0.0.1:%DASH%/
echo.
pause
endlocal
