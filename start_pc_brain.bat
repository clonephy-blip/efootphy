@echo off
chcp 437 >nul
title eFootball AI Assistant - PC Brain (Method B)
color 0b

echo =====================================================================
echo    eFootball Pro AI Assistant - PC Co-Processor (Method B)
echo =====================================================================
echo.

echo [1/3] Kiem tra ket noi USB Android...
"C:\scrcpy-win64-v4.1\adb.exe" devices
echo.

echo [2/3] Thiet lap kenh truyen du lieu USB Port 8888...
"C:\scrcpy-win64-v4.1\adb.exe" forward tcp:8888 tcp:8888
start /b "" node "%~dp0tools\pc_brain\bridge.js"
echo [OK] USB Port 8888 Bridge da khoi chay.
echo.

echo [3/3] Khoi chay AI Vision + Scrcpy Game Stream...
echo Ghi chu: Cua so game se hien thi. Du lieu HUD se ban vao dien thoai.
echo =====================================================================
echo.

"C:\scrcpy-win64-v4.1\scrcpy.exe" --window-title "eFootball-Screen" --always-on-top -m 640 --max-fps 30 --no-audio --record=- --record-format=mkv | wsl python3 /mnt/c/Users/PC/.gemini/antigravity/scratch/efootphy/tools/pc_brain/efootball_pc_brain.py

pause
