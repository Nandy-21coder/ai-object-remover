@echo off
if exist "C:\Users\user\anaconda3\Scripts\activate.bat" (
    call "C:\Users\user\anaconda3\Scripts\activate.bat"
)
cd /d "%~dp0"
python -m uvicorn app:app --host 127.0.0.1 --port 8000
