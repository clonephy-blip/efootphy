@echo off
chcp 65001 >nul
title eFootball AI Assistant - PC Brain (Method B)
color 0b

echo =====================================================================
echo    ⚽ eFootball Pro AI Assistant - PC Co-Processor (Method B) ⚽
echo =====================================================================
echo.
echo [1/3] Đang kiểm tra kết nối điện thoại qua USB...
C:\adb\platform-tools\adb.exe devices
echo.

echo [2/3] Thiết lập kênh truyền dữ liệu USB (Port 8888)...
C:\adb\platform-tools\adb.exe forward tcp:8888 tcp:8888
if %errorlevel% neq 0 (
    echo [CANH BAO] Khong the thiet lap adb forward. Vui long kiem tra cap USB!
) else (
    echo [OK] Kênh kết nối USB Port 8888 đã sẵn sàng!
)
echo.

echo [3/3] Khởi động màn hình truyền hình ảnh scrcpy (60 FPS)...
start "" "C:\scrcpy-win64-v4.1\scrcpy.exe" -m 1280 --max-fps 60 --always-on-top --window-title "eFootball-Screen" --no-audio
echo [OK] scrcpy đã bật!
echo.

echo =====================================================================
echo 🚀 Đang khởi chạy Trí tuệ nhân tạo (YOLO11n + Phân loại 2 đội + Chuyền)...
echo 💡 Giữ cửa sổ này chạy khi chơi game. Nhấn Ctrl+C để dừng.
echo =====================================================================
echo.

wsl python3 /mnt/c/Users/PC/.gemini/antigravity/scratch/efootphy/tools/pc_brain/efootball_pc_brain.py

pause
