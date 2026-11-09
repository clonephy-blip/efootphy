@echo off
setlocal
title eFootPhy - Cai dat PC lan dau
cd /d "%~dp0"
set "SETUP_RESULT=1"
echo Tao moi truong Python rieng cho eFootPhy...
if exist ".venv\Scripts\python.exe" goto install_deps
py -3.12 -c "import struct,sys; assert sys.version_info[:2]==(3,12) and struct.calcsize('P')==8" >nul 2>&1
if errorlevel 1 goto missing_python
py -3.12 -m venv .venv
if errorlevel 1 goto finish
:install_deps
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r requirements-pc-cpu.txt
if errorlevel 1 goto finish
".venv\Scripts\python.exe" -c "import cv2,mss,numpy,PIL,onnxruntime as ort; ort.disable_telemetry_events(); assert 'CPUExecutionProvider' in ort.get_available_providers(); print('DA CAI XONG - CPU runtime san sang')"
if errorlevel 1 goto finish
set "SETUP_RESULT=0"
echo.
echo Lan sau chi can mo 02_CHAY_TRO_LY.bat.
goto finish
:missing_python
echo Chua tim thay Python 3.12 64-bit ban day du.
echo Mo HUONG_DAN.txt de lay link cai Python; sau do chay lai file nay.
:finish
if "%SETUP_RESULT%"=="0" goto pause_if_needed
echo.
echo CAI DAT CHUA THANH CONG. Gui nguyen cac dong loi o tren.
:pause_if_needed
if not "%~1"=="--no-pause" pause
exit /b %SETUP_RESULT%
