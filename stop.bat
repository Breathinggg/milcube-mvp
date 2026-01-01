@echo off
echo [MilCube-MVP] Stopping all MilCube processes...

REM 关闭特定标题的窗口
taskkill /FI "WINDOWTITLE eq MilCube-cam5*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq MilCube-cam6*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq MilCube-dashboard*" /F >nul 2>&1

echo [MilCube-MVP] All MilCube processes stopped.
