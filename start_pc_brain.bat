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

echo [2/3] Thiet lap cong ket noi USB Port 8888...
"C:\scrcpy-win64-v4.1\adb.exe" forward tcp:8888 tcp:8888
echo [OK] USB Port 8888 da forward toi dien thoai.
echo.

echo [3/3] Khoi chay man hinh game scrcpy (60 FPS)...
start "" "C:\scrcpy-win64-v4.1\scrcpy.exe" --window-title "eFootball-Screen" --always-on-top -m 1280 --max-fps 60 --no-audio
echo [OK] Cua so game da bat!
echo.

echo =====================================================================
echo [4/4] Khoi chay AI Vision (YOLO11n + Phan loai 2 doi + Chuyen)...
echo =====================================================================
echo.

"C:\Users\PC\PythonEmbed\python.exe" "%~dp0tools\pc_brain\efootball_pc_brain.py"

pause
