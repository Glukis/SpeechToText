@echo off
chcp 65001 >nul
cd /d "%~dp0"
color 0A
cls

echo.
echo     ======================================================
echo                  SPEECH TO TEXT
echo     ======================================================
echo.
echo     Current directory: %CD%
echo.
echo     Starting program...
echo.

".venv\Scripts\python.exe" SpeechToText.py

if errorlevel 1 (
    echo.
    echo     ======================================================
    echo                  ERROR!
    echo     ======================================================
    echo.
    pause
)
