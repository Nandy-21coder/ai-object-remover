@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo AI Photo Studio - Launcher
echo ============================================================

if exist "..\.venv\Scripts\python.exe" (
    "..\.venv\Scripts\python.exe" run.py %*
) else if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" run.py %*
) else (
    python run.py %*
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Launcher encountered an error.
    pause
)
