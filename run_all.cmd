@echo off
setlocal
python -B -X utf8 "%~dp0run_all.py" %*
if errorlevel 1 (
    pause
    exit /b 1
)
exit /b 0
