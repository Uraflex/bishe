#!/bin/bash

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "========================================"
echo "  信用卡欺诈检测系统 v2.0 - 增强版"
echo "========================================"
echo

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}[错误]${NC} 未找到Python3，请先安装Python 3.7或更高版本"
    echo "Ubuntu/Debian: sudo apt install python3 python3-pip python3-venv"
    echo "CentOS/RHEL: sudo yum install python3 python3-pip"
    echo "macOS: brew install python3"
    exit 1
fi

echo -e "${GREEN}[信息]${NC} 检测到Python环境:"
python3 --version

# 检查虚拟环境
if [ ! -d ".venv" ]; then
    echo -e "${GREEN}[信息]${NC} 创建虚拟环境..."
    python3 -m venv .venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}[错误]${NC} 虚拟环境创建失败"
        exit 1
    fi
fi

# 激活虚拟环境
echo -e "${GREEN}[信息]${NC} 激活虚拟环境..."
source .venv/bin/activate
if [ $? -ne 0 ]; then
    echo -e "${RED}[错误]${NC} 虚拟环境激活失败"
    exit 1
fi

# 检查requirements.txt是否存在
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}[错误]${NC} 未找到requirements.txt文件"
    exit 1
fi

# 安装依赖
echo -e "${GREEN}[信息]${NC} 安装/更新依赖包..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}[警告]${NC} 依赖包安装可能存在问题，但继续运行..."
fi

# 检查主程序文件是否存在
if [ ! -f "enhanced_gui.py" ]; then
    echo -e "${RED}[错误]${NC} 未找到enhanced_gui.py文件"
    exit 1
fi

# 启动增强版GUI
echo
echo -e "${GREEN}[信息]${NC} 启动增强版图形界面..."
echo -e "${BLUE}[提示]${NC} 程序启动可能需要几秒钟，请耐心等待..."
echo
python enhanced_gui.py

# 检查程序退出状态
if [ $? -ne 0 ]; then
    echo
    echo -e "${RED}[错误]${NC} 程序运行出现错误"
    echo -e "${BLUE}[建议]${NC} 请检查数据文件是否存在或查看上方错误信息"
fi

echo
echo -e "${GREEN}[信息]${NC} 程序已退出"
