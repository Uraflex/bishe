@echo off
chcp 65001 >nul
title Credit Card Fraud Detection v2.0

echo ========================================
echo   Credit Card Fraud Detection v2.0
echo ========================================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found, please install Python 3.7+
    echo Download: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo INFO: Python environment detected:
python --version

if not exist "requirements.txt" (
    echo ERROR: requirements.txt not found
    pause
    exit /b 1
)

echo INFO: Installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo WARNING: Dependencies installation may have issues, but continuing...
)

if not exist "enhanced_gui.py" (
    echo ERROR: enhanced_gui.py not found
    pause
    exit /b 1
)

echo.
echo INFO: Starting GUI...
echo TIP: Program startup may take a few seconds, please wait...
echo.
python enhanced_gui.py

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Program failed with error code: %errorlevel%
    echo SUGGESTION: Check if data files exist or review error messages above
)

echo.
echo INFO: Program exited
pause
