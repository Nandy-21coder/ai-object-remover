@echo off
setlocal
cd /d "%~dp0"

:: Check if server is already running on port 8000
curl.exe -s -m 1 http://127.0.0.1:8000/api/health >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    start /min "" cmd /c ""C:\Users\user\anaconda3\Scripts\activate.bat" && cd /d "%~dp0backend" && python -m uvicorn app:app --host 127.0.0.1 --port 8000"
    timeout /t 2 /nobreak >nul
)

:: If run with /autostart flag (from Windows Startup folder), do not open browser
if /i "%1"=="/autostart" goto end

:: Open the app directly in user's default browser
start http://127.0.0.1:8000/

:end
