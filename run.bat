@echo off
setlocal
title ATG Self-Healing Classification Project
echo ============================================
echo        ATG Self-Healing Classification
echo ============================================
echo.

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "VENV_PY=%SCRIPT_DIR%\venv\Scripts\python.exe"

where python >nul 2>nul
if %errorlevel% neq 0 (
  echo [ERROR] Python not found in PATH. Please install Python 3.8+ and retry.
  pause
  exit /b 1
)

if not exist "%SCRIPT_DIR%\venv" (
    echo [SETUP] Creating virtual environment...
    python -m venv "%SCRIPT_DIR%\venv"
)

if not exist "%VENV_PY%" (
    echo [ERROR] venv Python not found at %VENV_PY%
    pause
    exit /b 1
)

echo [SETUP] Installing dependencies...
"%VENV_PY%" -m pip install --upgrade pip
"%VENV_PY%" -m pip install -r "%SCRIPT_DIR%\requirements.txt"

if not exist "%SCRIPT_DIR%\models\backup_model.joblib" (
    echo [TRAIN] Training backup model...
    "%VENV_PY%" "%SCRIPT_DIR%\train_backup_model.py"
) else (
    echo [INFO] Backup model already exists. Skipping training.
)

echo.
echo ============================================
echo [RUN] Starting the Self-Healing CLI
echo Type your text below (type 'exit' to quit)
echo ============================================
echo.
"%VENV_PY%" -u "%SCRIPT_DIR%\cli.py"

echo.
echo [DONE] Program finished.
pause
endlocal
