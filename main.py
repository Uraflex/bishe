#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信用卡欺诈检测系统 - 程序入口
启动GUI启动器让用户选择运行模式
"""

import sys

def main():
    """主函数 - 启动GUI启动器"""
    try:
        from launcher import LauncherGUI
        import tkinter as tk
        
        root = tk.Tk()
        app = LauncherGUI(root)
        root.mainloop()
        
    except ImportError as e:
        print(f"错误：无法导入启动器模块 - {e}")
        print("请确保 launcher.py 文件存在")
        sys.exit(1)
    except Exception as e:
        print(f"启动时发生错误：{e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
