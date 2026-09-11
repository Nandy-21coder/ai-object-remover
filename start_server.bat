@echo off
title AI Object Remover Server

echo ===================================================
echo   Starting AI Object Remover Backend Server...
echo ===================================================

if exist "C:\Users\user\anaconda3\Scripts\activate.bat" (
    call "C:\Users\user\anaconda3\Scripts\activate.bat"
)
cd /d "%~dp0backend"
python -m uvicorn app:app --host 127.0.0.1 --port 8000

pause
