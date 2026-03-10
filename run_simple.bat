@echo off
title Credit Card Fraud Detection - Quick Start

echo ========================================
echo   Credit Card Fraud Detection - Quick Start
echo ========================================
echo.

echo Starting GUI...
python enhanced_gui.py

if %errorlevel% neq 0 (
    echo.
    echo Startup failed! Please ensure:
    echo 1. Python 3.7+ is installed
    echo 2. Dependencies are installed: pip install -r requirements.txt
    echo 3. enhanced_gui.py file exists
    echo.
)

pause
