@echo off
chcp 65001 >nul
cd /d "%~dp0"
cls

echo ======================================================
echo           Creating EXE file
echo ======================================================
echo.
echo WARNING: EXE build for Whisper is COMPLEX:
echo - Final EXE will be LARGE (100+ MB)
echo - Whisper models will download on first run
echo - Requires PyTorch and all dependencies
echo.
echo It's RECOMMENDED to use .bat file instead!
echo Press Ctrl+C to cancel or
pause
echo.

REM Check if PyInstaller is installed
".venv\Scripts\python.exe" -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo Installing PyInstaller...
    ".venv\Scripts\pip.exe" install pyinstaller
    echo.
)

echo Creating EXE with PyInstaller...
echo This may take several minutes...
echo.

REM Build the EXE with console (so users can see what's happening)
".venv\Scripts\pyinstaller.exe" --onefile --name="SpeechToText" ^
    --hidden-import=whisper ^
    --hidden-import=torch ^
    --hidden-import=torchaudio ^
    --hidden-import=numpy ^
    --hidden-import=pyaudio ^
    --hidden-import=pynput ^
    --hidden-import=pyperclip ^
    --hidden-import=keyboard ^
    --hidden-import=pyautogui ^
    --collect-all whisper ^
    --collect-all torch ^
    voice_recorder.py

if errorlevel 1 (
    echo.
    echo ======================================================
    echo                  ERROR!
    echo ======================================================
    echo Failed to create EXE
    echo Check the error messages above
    echo.
    pause
    exit /b 1
)

echo.
echo ======================================================
echo                  SUCCESS!
echo ======================================================
echo.
echo EXE file created: dist\VoiceRecorder.exe
echo.
echo IMPORTANT NOTES:
echo - First run will download Whisper model (takes time)
echo - EXE is large due to PyTorch and dependencies
echo - Antivirus may flag it (false positive)
echo.
echo To use:
echo 1. Copy dist\VoiceRecorder.exe to desired location
echo 2. Run it (may need admin rights for mouse hook)
echo 3. Wait for Whisper model download on first run
echo.
pause
