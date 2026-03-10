@echo off
chcp 65001 >nul
title 信用卡欺诈检测系统 v2.0 - 增强版

echo ========================================
echo   信用卡欺诈检测系统 v2.0 - 增强版
echo ========================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到Python，请先安装Python 3.7或更高版本
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [信息] 检测到Python环境:
python --version

REM 检查虚拟环境
if not exist ".venv" (
    echo [信息] 创建虚拟环境...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [错误] 虚拟环境创建失败
        pause
        exit /b 1
    )
)

REM 激活虚拟环境
echo [信息] 激活虚拟环境...
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [错误] 虚拟环境激活失败
    pause
    exit /b 1
)

REM 检查requirements.txt是否存在
if not exist "requirements.txt" (
    echo [错误] 未找到requirements.txt文件
    pause
    exit /b 1
)

REM 安装依赖
echo [信息] 安装/更新依赖包...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [警告] 依赖包安装可能存在问题，但继续运行...
)

REM 检查主程序文件是否存在
if not exist "enhanced_gui.py" (
    echo [错误] 未找到enhanced_gui.py文件
    pause
    exit /b 1
)

REM 启动增强版GUI
echo.
echo [信息] 启动增强版图形界面...
echo [提示] 程序启动可能需要几秒钟，请耐心等待...
echo.
python enhanced_gui.py

REM 检查程序退出状态
if %errorlevel% neq 0 (
    echo.
    echo [错误] 程序运行出现错误，错误代码: %errorlevel%
    echo [建议] 请检查数据文件是否存在或查看上方错误信息
)

echo.
echo [信息] 程序已退出
pause
