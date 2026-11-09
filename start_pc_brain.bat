@echo off
setlocal
title eFootPhy PC Brain
cd /d "%~dp0"

if not defined EFOOTPHY_SCRCPY_DIR set "EFOOTPHY_SCRCPY_DIR=C:\scrcpy-win64-v4.1"
if not defined EFOOTPHY_PYTHON set "EFOOTPHY_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%EFOOTPHY_SCRCPY_DIR%\adb.exe" goto missing_tools
if not exist "%EFOOTPHY_SCRCPY_DIR%\scrcpy.exe" goto missing_tools
if not exist "%EFOOTPHY_PYTHON%" (
    echo Python chua duoc cau hinh. Doc docs\handover\RUN_PC.md.
    goto failed
)
if not defined EFOOTPHY_FIELD_MODEL (
    echo Can dat EFOOTPHY_FIELD_MODEL tro den model FIELD .onnx tren PC.
    echo Khong dung model minimap hay pitch cho duong dan nay.
    goto failed
)
if not exist "%EFOOTPHY_FIELD_MODEL%" (
    echo Khong tim thay model: "%EFOOTPHY_FIELD_MODEL%"
    goto failed
)

echo [1/3] Kiem tra dung mot dien thoai USB da cho phep debugging...
"%EFOOTPHY_SCRCPY_DIR%\adb.exe" -d get-state
if errorlevel 1 goto failed
"%EFOOTPHY_SCRCPY_DIR%\adb.exe" -d forward tcp:8888 tcp:8888
if errorlevel 1 goto failed

echo [2/3] Mo scrcpy chi doc. Hay dong cua so eFootball-Screen cu truoc.
start "" "%EFOOTPHY_SCRCPY_DIR%\scrcpy.exe" --window-title "eFootball-Screen" --always-on-top -m 1920 --max-fps 60 --no-audio --no-control --select-usb
timeout /t 3 /nobreak >nul
echo [3/3] Bat PC AI. Android can bat HUD trong ung dung.
"%EFOOTPHY_PYTHON%" "%~dp0tools\pc_brain\efootball_pc_brain.py"
set "EFOOTPHY_EXIT=%errorlevel%"
pause
exit /b %EFOOTPHY_EXIT%

:missing_tools
echo Khong tim thay adb.exe va scrcpy.exe trong "%EFOOTPHY_SCRCPY_DIR%".
echo Dat EFOOTPHY_SCRCPY_DIR thanh thu muc scrcpy cua ban.
:failed
pause
exit /b 1
