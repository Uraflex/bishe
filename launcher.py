#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信用卡欺诈检测系统 - GUI启动器
提供图形化界面选择运行模式
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import sys
import subprocess
import os


class LauncherGUI:
    """启动器GUI类"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("信用卡欺诈检测系统 - 启动器")
        self.root.geometry("600x500")
        self.root.resizable(False, False)
        
        # 设置窗口居中
        self.center_window()
        
        # 配色方案
        self.colors = {
            'bg': '#f5f5f5',
            'primary': '#2196F3',
            'secondary': '#4CAF50',
            'accent': '#FF9800',
            'text': '#333333',
            'light_text': '#666666'
        }
        
        self.root.configure(bg=self.colors['bg'])
        
        self.create_widgets()
    
    def center_window(self):
        """将窗口居中显示"""
        self.root.update_idletasks()
        width = 600
        height = 500
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def create_widgets(self):
        """创建界面组件"""
        # 标题区域
        title_frame = tk.Frame(self.root, bg=self.colors['bg'])
        title_frame.pack(fill=tk.X, padx=20, pady=20)
        
        title_label = tk.Label(
            title_frame,
            text="信用卡欺诈检测系统",
            font=('Microsoft YaHei', 24, 'bold'),
            fg=self.colors['primary'],
            bg=self.colors['bg']
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            title_frame,
            text="请选择运行模式",
            font=('Microsoft YaHei', 12),
            fg=self.colors['light_text'],
            bg=self.colors['bg']
        )
        subtitle_label.pack(pady=5)
        
        # 分隔线
        separator = ttk.Separator(self.root, orient='horizontal')
        separator.pack(fill=tk.X, padx=20, pady=10)
        
        # 按钮区域
        buttons_frame = tk.Frame(self.root, bg=self.colors['bg'])
        buttons_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=10)
        
        # GUI模式按钮
        self.create_mode_button(
            buttons_frame,
            "GUI 界面",
            "图形化操作界面",
            "适合：数据分析、可视化展示、交互式操作",
            "🖥️",
            self.colors['secondary'],
            self.launch_gui
        )
        
        # 间距
        tk.Frame(buttons_frame, height=15, bg=self.colors['bg']).pack()
        
        # CLI模式按钮
        self.create_mode_button(
            buttons_frame,
            "CLI 界面",
            "命令行界面",
            "适合：批量实验、参数调优、自动化脚本",
            "⌨️",
            self.colors['accent'],
            self.launch_cli
        )
        
        # 底部区域
        bottom_frame = tk.Frame(self.root, bg=self.colors['bg'])
        bottom_frame.pack(fill=tk.X, padx=20, pady=20)
        
        # 退出按钮
        exit_btn = tk.Button(
            bottom_frame,
            text="退出",
            font=('Microsoft YaHei', 11),
            bg='#e0e0e0',
            fg=self.colors['text'],
            activebackground='#d0d0d0',
            width=12,
            cursor='hand2',
            relief=tk.FLAT,
            command=self.on_exit
        )
        exit_btn.pack(side=tk.RIGHT)
        
        # 版本信息
        version_label = tk.Label(
            bottom_frame,
            text="v1.0 | 信用卡欺诈检测系统",
            font=('Microsoft YaHei', 9),
            fg=self.colors['light_text'],
            bg=self.colors['bg']
        )
        version_label.pack(side=tk.LEFT)
    
    def create_mode_button(self, parent, title, subtitle, desc, icon, color, command):
        """创建模式选择按钮"""
        btn_frame = tk.Frame(
            parent,
            bg='white',
            highlightbackground='#ddd',
            highlightthickness=1
        )
        btn_frame.pack(fill=tk.X, pady=5)
        
        # 鼠标悬停效果
        def on_enter(e):
            btn_frame.configure(bg='#fafafa')
            inner_frame.configure(bg='#fafafa')
            
        def on_leave(e):
            btn_frame.configure(bg='white')
            inner_frame.configure(bg='white')
        
        btn_frame.bind('<Enter>', on_enter)
        btn_frame.bind('<Leave>', on_leave)
        
        inner_frame = tk.Frame(btn_frame, bg='white', padx=15, pady=15)
        inner_frame.pack(fill=tk.X)
        inner_frame.bind('<Enter>', on_enter)
        inner_frame.bind('<Leave>', on_leave)
        
        # 点击事件
        for widget in [btn_frame, inner_frame]:
            widget.bind('<Button-1>', lambda e: command())
            widget.config(cursor='hand2')
        
        # 左侧彩色条
        color_bar = tk.Frame(inner_frame, bg=color, width=5)
        color_bar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        
        # 内容区域
        content_frame = tk.Frame(inner_frame, bg='white')
        content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 标题行
        title_frame = tk.Frame(content_frame, bg='white')
        title_frame.pack(fill=tk.X)
        
        icon_label = tk.Label(
            title_frame,
            text=icon,
            font=('Segoe UI Emoji', 24),
            bg='white'
        )
        icon_label.pack(side=tk.LEFT)
        icon_label.bind('<Button-1>', lambda e: command())
        
        text_frame = tk.Frame(title_frame, bg='white', padx=10)
        text_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        title_text = tk.Label(
            text_frame,
            text=title,
            font=('Microsoft YaHei', 16, 'bold'),
            fg=color,
            bg='white'
        )
        title_text.pack(anchor='w')
        title_text.bind('<Button-1>', lambda e: command())
        
        subtitle_text = tk.Label(
            text_frame,
            text=subtitle,
            font=('Microsoft YaHei', 10),
            fg=self.colors['light_text'],
            bg='white'
        )
        subtitle_text.pack(anchor='w')
        subtitle_text.bind('<Button-1>', lambda e: command())
        
        # 描述
        desc_label = tk.Label(
            content_frame,
            text=desc,
            font=('Microsoft YaHei', 9),
            fg=self.colors['light_text'],
            bg='white',
            wraplength=350,
            justify=tk.LEFT
        )
        desc_label.pack(anchor='w', pady=(5, 0))
        desc_label.bind('<Button-1>', lambda e: command())
        
        # 右侧箭头
        arrow_label = tk.Label(
            inner_frame,
            text="›",
            font=('Microsoft YaHei', 24),
            fg='#ccc',
            bg='white'
        )
        arrow_label.pack(side=tk.RIGHT)
        arrow_label.bind('<Button-1>', lambda e: command())
    
    def launch_gui(self):
        """启动GUI界面"""
        self.root.destroy()
        try:
            from enhanced_gui import EnhancedCreditCardFraudGUI
            root = tk.Tk()
            app = EnhancedCreditCardFraudGUI(root)
            root.mainloop()
        except ImportError as e:
            messagebox.showerror("错误", f"无法导入GUI模块: {e}")
            sys.exit(1)
        except Exception as e:
            messagebox.showerror("错误", f"启动GUI时发生错误: {e}")
            sys.exit(1)
    
    def launch_cli(self):
        """启动CLI界面"""
        self.show_cli_dialog()
    
    def show_cli_dialog(self):
        """显示CLI参数输入对话框"""
        dialog = tk.Toplevel(self.root)
        dialog.title("CLI 命令行界面")
        dialog.geometry("600x450")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # 居中显示
        dialog.update_idletasks()
        width = 600
        height = 450
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (width // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (height // 2)
        dialog.geometry(f'{width}x{height}+{x}+{y}')
        
        # 说明区域
        tk.Label(
            dialog,
            text="命令行界面 (CLI) - 适合批量实验和自动化脚本",
            font=('Microsoft YaHei', 14, 'bold'),
            fg='#333'
        ).pack(pady=10)
        
        # 使用示例
        example_frame = tk.LabelFrame(dialog, text="使用示例", padx=10, pady=10)
        example_frame.pack(fill=tk.X, padx=20, pady=5)
        
        examples = [
            "基础训练:  python cli.py -d ./data/creditcard.csv -t",
            "训练+调优:  python cli.py -d ./data/creditcard.csv -t -u",
            "完整参数:  python cli.py -d ./data/creditcard.csv -t -u -r 3.3 -s ./model/best.pkl"
        ]
        
        for ex in examples:
            tk.Label(example_frame, text=ex, font=('Consolas', 9), fg='#666').pack(anchor='w')
        
        # 快速选择区域
        quick_frame = tk.LabelFrame(dialog, text="快速启动", padx=10, pady=10)
        quick_frame.pack(fill=tk.X, padx=20, pady=10)
        
        quick_btn_frame = tk.Frame(quick_frame)
        quick_btn_frame.pack()
        
        def run_cli_with_args(args):
            self.root.destroy()
            try:
                cmd = [sys.executable, 'cli.py'] + args.split()
                subprocess.run(cmd)
            except Exception as e:
                messagebox.showerror("错误", f"运行CLI时出错: {e}")
                sys.exit(1)
            sys.exit(0)
        
        tk.Button(
            quick_btn_frame,
            text="查看帮助",
            command=lambda: run_cli_with_args('-h'),
            width=15
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            quick_btn_frame,
            text="基础训练",
            command=lambda: run_cli_with_args('-d ./data/creditcard.csv -t'),
            width=15,
            bg='#4CAF50',
            fg='white'
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            quick_btn_frame,
            text="训练+调优",
            command=lambda: run_cli_with_args('-d ./data/creditcard.csv -t -u'),
            width=15,
            bg='#2196F3',
            fg='white'
        ).pack(side=tk.LEFT, padx=5)
        
        # 自定义参数
        custom_frame = tk.LabelFrame(dialog, text="自定义参数", padx=10, pady=10)
        custom_frame.pack(fill=tk.X, padx=20, pady=5)
        
        tk.Label(custom_frame, text="输入参数:").pack(anchor='w')
        
        args_entry = tk.Entry(custom_frame, width=60, font=('Consolas', 10))
        args_entry.pack(fill=tk.X, pady=5)
        args_entry.insert(0, "-d ./data/creditcard.csv -t -u -r 3.3")
        
        def run_custom():
            args = args_entry.get().strip()
            if args:
                run_cli_with_args(args)
        
        tk.Button(
            custom_frame,
            text="运行",
            command=run_custom,
            bg='#FF9800',
            fg='white',
            width=10
        ).pack(anchor='e', pady=5)
        
        # 底部按钮
        btn_frame = tk.Frame(dialog)
        btn_frame.pack(fill=tk.X, padx=20, pady=15)
        
        tk.Button(
            btn_frame,
            text="返回",
            command=dialog.destroy,
            width=12
        ).pack(side=tk.RIGHT)
    
    def on_exit(self):
        """退出程序"""
        if messagebox.askyesno("确认", "确定要退出吗?"):
            self.root.destroy()
            sys.exit(0)


def main():
    """主函数"""
    root = tk.Tk()
    app = LauncherGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
