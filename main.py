#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tkinter as tk
import sys

def main():
    try:
        from enhanced_gui import EnhancedCreditCardFraudGUI
        root = tk.Tk()
        app = EnhancedCreditCardFraudGUI(root)
        root.mainloop()
    except ImportError as e:
        print(f"错误：无法导入GUI模块 - {e}")
        print("请确保 enhanced_gui.py 文件存在")
        sys.exit(1)
    except Exception as e:
        print(f"启动GUI时发生错误：{e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
