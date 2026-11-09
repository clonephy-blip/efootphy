@echo off
setlocal
title eFootPhy - Chay tro ly PC
cd /d "%~dp0"
set "EFOOTPHY_PYTHON=%~dp0.venv\Scripts\python.exe"
set "EFOOTPHY_FIELD_MODEL=%~dp0models\field640.onnx"
set "EFOOTPHY_EXECUTION_PROVIDER=cpu"
set "EFOOTPHY_SHADOW_EVIDENCE=0"
if not defined EFOOTPHY_SCRCPY_DIR set "EFOOTPHY_SCRCPY_DIR=C:\scrcpy-win64-v4.1"
call "%~dp0start_pc_brain.bat"
exit /b %errorlevel%
