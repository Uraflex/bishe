#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns
import threading
import os
from datetime import datetime

from enhanced_data_processor import EnhancedDataProcessor
from model_trainer import ModelTrainer
from evaluator import ModelEvaluator
from utils import (plot_confusion_matrix, plot_roc_curve, plot_feature_importance,
                   create_summary_report)
from hyperparameter_optimizer import HybridOptimizer
from model_comparator import ModelComparator

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class EnhancedCreditCardFraudGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("信用卡欺诈检测系统 - 增强版")
        self.root.geometry("1200x750")
        
        # 设置样式
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # 初始化变量
        self.data_file_path = tk.StringVar()
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_enhanced = None
        self.y_train_enhanced = None
        self.evaluation_results = {}
        self.feature_importance_df = None
        self.comparison_results = None
        self.ablation_results = None
        
        # 初始化处理器
        self.processor = EnhancedDataProcessor()
        self.enhanced_processor = EnhancedDataProcessor()
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()
        self.optimizer = HybridOptimizer(max_iter=50)  # 减少迭代次数以加快速度
        self.comparator = ModelComparator(cv_folds=3)  # 减少折数以加快速度
        
        # 创建界面
        self.create_widgets()
        
    def create_widgets(self):
        """创建增强界面组件"""
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)
        
        # 标题
        title_label = ttk.Label(main_frame, text="信用卡欺诈检测系统", 
                               font=('Arial', 18, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=10)
        
        # 左侧控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding="8")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # 数据加载部分
        data_frame = ttk.LabelFrame(control_frame, text="数据模块", padding="4")
        data_frame.pack(fill=tk.X, pady=5)
        
        ttk.Entry(data_frame, textvariable=self.data_file_path, width=35).pack(side=tk.LEFT, padx=5)
        ttk.Button(data_frame, text="浏览", command=self.browse_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(data_frame, text="加载数据", command=self.load_data).pack(side=tk.LEFT, padx=2)
        
        # 模型模块
        model_frame = ttk.LabelFrame(control_frame, text="模型模块", padding="4")
        model_frame.pack(fill=tk.X, pady=3)
        
        ttk.Button(model_frame, text="训练模型 (XGBoost)", command=self.train_enhanced_model).pack(fill=tk.X, pady=1)
        ttk.Button(model_frame, text="模型评估", command=self.evaluate_model).pack(fill=tk.X, pady=1)
        ttk.Button(model_frame, text="性能对比", command=self.compare_models).pack(fill=tk.X, pady=1)
        ttk.Button(model_frame, text="模型导出", command=self.save_model).pack(fill=tk.X, pady=1)
        
        # 结果模块
        result_frame = ttk.LabelFrame(control_frame, text="结果模块", padding="4")
        result_frame.pack(fill=tk.X, pady=3)
        
        ttk.Button(result_frame, text="可视化分析", command=self.show_visualization_menu).pack(fill=tk.X, pady=1)
        ttk.Button(result_frame, text="特征重要性", command=self.plot_feature_importance).pack(fill=tk.X, pady=1)
        ttk.Button(result_frame, text="消融实验", command=self.ablation_study).pack(fill=tk.X, pady=1)
        ttk.Button(result_frame, text="对比结果图表", command=self.plot_comparison_results).pack(fill=tk.X, pady=1)
        ttk.Button(result_frame, text="导出结果", command=self.export_results).pack(fill=tk.X, pady=1)
        
        # 进度条
        self.progress = ttk.Progressbar(control_frame, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=10)
        
        # 状态标签xunl
        self.status_label = ttk.Label(control_frame, text="就绪", relief=tk.SUNKEN)
        self.status_label.pack(fill=tk.X, pady=5)
        
        # 右侧结果显示面板
        result_frame = ttk.LabelFrame(main_frame, text="结果显示", padding="10")
        result_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # 创建笔记本控件用于多标签页
        self.notebook = ttk.Notebook(result_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # 数据信息标签页
        self.data_info_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.data_info_frame, text="数据信息")
        
        self.data_info_text = tk.Text(self.data_info_frame, wrap=tk.WORD, height=20)
        data_scrollbar = ttk.Scrollbar(self.data_info_frame, orient=tk.VERTICAL, command=self.data_info_text.yview)
        self.data_info_text.configure(yscrollcommand=data_scrollbar.set)
        self.data_info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        data_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 模型评估结果标签页
        self.eval_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.eval_frame, text="评估结果")
        
        self.eval_text = tk.Text(self.eval_frame, wrap=tk.WORD, height=20)
        eval_scrollbar = ttk.Scrollbar(self.eval_frame, orient=tk.VERTICAL, command=self.eval_text.yview)
        self.eval_text.configure(yscrollcommand=eval_scrollbar.set)
        self.eval_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        eval_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 模型对比结果标签页
        self.compare_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.compare_frame, text="模型对比")
        
        self.compare_text = tk.Text(self.compare_frame, wrap=tk.WORD, height=20)
        compare_scrollbar = ttk.Scrollbar(self.compare_frame, orient=tk.VERTICAL, command=self.compare_text.yview)
        self.compare_text.configure(yscrollcommand=compare_scrollbar.set)
        self.compare_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        compare_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 图表显示标签页
        self.plot_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.plot_frame, text="图表")
        
        # 创建matplotlib图形
        self.figure = Figure(figsize=(12, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def switch_to_plot_tab(self):
        """安全地切换到图表选项卡"""
        try:
            # 获取选项卡总数
            tab_count = len(self.notebook.tabs())
            # 图表选项卡总是最后一个（索引从0开始）
            plot_tab_index = tab_count - 1
            # 确保索引有效
            if 0 <= plot_tab_index < tab_count:
                self.notebook.select(plot_tab_index)
        except Exception as e:
            # 如果切换失败，静默处理不影响绘图
            pass
    
    def browse_file(self):
        """浏览文件对话框"""
        filename = filedialog.askopenfilename(
            title="选择信用卡数据文件",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]
        )
        if filename:
            self.data_file_path.set(filename)
            
    def update_status(self, message):
        """更新状态标签"""
        self.status_label.config(text=message)
        self.root.update_idletasks()
        
    def start_progress(self):
        """开始进度条动画"""
        self.progress.start(10)
        
    def stop_progress(self):
        """停止进度条动画"""
        self.progress.stop()
        
    def load_data(self):
        """加载数据（基础版本）"""
        if not self.data_file_path.get():
            messagebox.showerror("错误", "请先选择数据文件")
            return
            
        def load_data_thread():
            try:
                self.start_progress()
                self.update_status("正在加载数据...")
                
                # 加载数据
                df = self.processor.load_data(self.data_file_path.get())
                if df is None:
                    return
                    
                # 探索数据
                self.update_status("正在分析数据...")
                
                # 清空文本框并显示数据信息
                self.data_info_text.delete(1.0, tk.END)
                
                # 基本信息输出
                info_text = f"基础数据加载完成！\n"
                info_text += f"{'='*50}\n\n"
                info_text += f"数据形状: {df.shape}\n"
                info_text += f"列名: {list(df.columns)}\n\n"
                
                # 类别分布
                if 'Class' in df.columns:
                    class_counts = df['Class'].value_counts()
                    class_percentages = df['Class'].value_counts(normalize=True) * 100
                    
                    info_text += "类别分布分析:\n"
                    for class_label in class_counts.index:
                        count = class_counts[class_label]
                        percentage = class_percentages[class_label]
                        class_name = "正常交易" if class_label == 0 else "欺诈交易"
                        info_text += f"- {class_name} (Class={class_label}): {count} 条 ({percentage:.4f}%)\n"
                
                # 预处理数据（使用正确的无数据泄露流程）
                self.update_status("正在预处理数据（无数据泄露）...")
                self.X_train, self.X_test, self.y_train, self.y_test = \
                    self.processor.preprocess_data_split(
                        df, 
                        test_size=0.2,
                        use_hybrid_sampling=False,  # 基础版本不采样
                        use_advanced_features=False,  # 基础版本无高级特征
                        random_state=42
                    )
                
                if self.X_train is not None:
                    # 训练集类别分布
                    train_class_counts = pd.Series(self.y_train).value_counts().sort_index()
                    info_text += f"\n基础预处理完成（无数据泄露）！\n"
                    info_text += "训练集类别分布:\n"
                    for class_label in train_class_counts.index:
                        count = train_class_counts[class_label]
                        class_name = "正常交易" if class_label == 0 else "欺诈交易"
                        info_text += f"- {class_name} (Class={class_label}): {count} 条\n"
                    
                    # 测试集类别分布
                    test_class_counts = pd.Series(self.y_test).value_counts().sort_index()
                    info_text += f"\n测试集类别分布:\n"
                    for class_label in test_class_counts.index:
                        count = test_class_counts[class_label]
                        percentage = count / len(self.y_test) * 100
                        class_name = "正常交易" if class_label == 0 else "欺诈交易"
                        info_text += f"- {class_name} (Class={class_label}): {count} 条 ({percentage:.2f}%)\n"
                    
                    info_text += f"\n训练集大小: {self.X_train.shape}\n"
                    info_text += f"测试集大小: {self.X_test.shape}\n"
                
                self.data_info_text.insert(tk.END, info_text)
                self.update_status("基础数据加载完成")
                
            except Exception as e:
                messagebox.showerror("错误", f"数据加载失败: {str(e)}")
                self.update_status("数据加载失败")
            finally:
                self.stop_progress()
        
        thread = threading.Thread(target=load_data_thread)
        thread.daemon = True
        thread.start()
        
    
    def train_basic_model(self):
        """训练基础模型"""
        if self.X_train is None:
            messagebox.showerror("错误", "请先加载数据")
            return
            
        def train_basic_model_thread():
            try:
                self.start_progress()
                self.update_status("正在训练基础模型...")
                
                # 训练XGBoost模型
                self.model = self.trainer.train_xgboost(self.X_train, self.y_train)
                
                self.update_status("基础模型训练完成")
                messagebox.showinfo("成功", "基础模型训练完成！")
                
            except Exception as e:
                messagebox.showerror("错误", f"基础模型训练失败: {str(e)}")
                self.update_status("基础模型训练失败")
            finally:
                self.stop_progress()
        
        thread = threading.Thread(target=train_basic_model_thread)
        thread.daemon = True
        thread.start()
    
    def optimize_hyperparameters(self):
        """超参数优化"""
        if self.X_train is None:
            messagebox.showerror("错误", "请先加载数据")
            return
            
        def optimize_thread():
            try:
                self.start_progress()
                self.update_status("正在进行超参数优化...")
                
                # 使用混合优化器
                best_params = self.optimizer.optimize(self.X_train, self.y_train, verbose=True)
                
                # 使用最佳参数训练模型
                self.update_status("使用最佳参数训练模型...")
                self.model = self.trainer.train_xgboost(self.X_train, self.y_train, **best_params)
                
                # 显示优化结果
                self.eval_text.delete(1.0, tk.END)
                result_text = f"超参数优化完成！\n"
                result_text += f"{'='*50}\n\n"
                result_text += "最佳参数配置:\n"
                for param, value in best_params.items():
                    result_text += f"  {param}: {value}\n"
                
                self.eval_text.insert(tk.END, result_text)
                
                self.update_status("超参数优化完成")
                messagebox.showinfo("成功", "超参数优化完成！")
                
            except Exception as e:
                messagebox.showerror("错误", f"超参数优化失败: {str(e)}")
                self.update_status("超参数优化失败")
            finally:
                self.stop_progress()
        
        thread = threading.Thread(target=optimize_thread)
        thread.daemon = True
        thread.start()
    
    def train_enhanced_model(self):
        """训练增强模型"""
        if not self.data_file_path.get():
            messagebox.showerror("错误", "请先选择数据文件")
            return
            
        def train_enhanced_model_thread():
            try:
                self.start_progress()
                self.update_status("正在训练模型...")
                
                # 加载并预处理数据
                df = self.enhanced_processor.load_data(self.data_file_path.get())
                if df is None:
                    return
                
                # 使用正确的预处理流程（先分割再处理，防止数据泄露）
                self.X_train_enhanced, self.X_test_enhanced, self.y_train_enhanced, self.y_test_enhanced = \
                    self.enhanced_processor.preprocess_data_split(
                        df, 
                        test_size=0.2,
                        use_hybrid_sampling=True, 
                        use_advanced_features=True, 
                        target_ratio=5,
                        random_state=42
                    )
                
                # 计算类别权重
                class_counts = pd.Series(self.y_train_enhanced).value_counts()
                scale_pos_weight = class_counts[0] / class_counts[1]
                
                # 训练增强XGBoost模型
                params = {
                    'scale_pos_weight': scale_pos_weight,
                    'random_state': 42
                }
                
                self.model = self.trainer.train_xgboost(self.X_train_enhanced, self.y_train_enhanced, **params)
                
                # 设置测试数据为增强版本
                self.X_test = self.X_test_enhanced
                self.y_test = self.y_test_enhanced
                
                self.update_status("增强模型训练完成")
                messagebox.showinfo("成功", "增强模型训练完成！")
                
            except Exception as e:
                messagebox.showerror("错误", f"增强模型训练失败: {str(e)}")
                self.update_status("增强模型训练失败")
            finally:
                self.stop_progress()
        
        thread = threading.Thread(target=train_enhanced_model_thread)
        thread.daemon = True
        thread.start()
    
    def evaluate_model(self):
        """评估模型"""
        if self.model is None:
            messagebox.showerror("错误", "请先训练模型")
            return
            
        def evaluate_model_thread():
            try:
                self.start_progress()
                self.update_status("正在评估模型...")
                
                # 预测
                y_pred = self.model.predict(self.X_test)
                y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
                
                # 计算评估指标
                accuracy = self.evaluator.calculate_accuracy(self.y_test, y_pred)
                report = self.evaluator.classification_report(self.y_test, y_pred)
                cm = self.evaluator.confusion_matrix(self.y_test, y_pred)
                fpr, tpr, roc_auc = self.evaluator.calculate_roc_auc(self.y_test, y_pred_proba)
                
                # 获取特征重要性
                feature_names = self.X_test.columns.tolist()
                self.feature_importance_df = self.evaluator.get_feature_importance(self.model, feature_names)
                
                # 保存评估结果
                self.evaluation_results = {
                    'accuracy': accuracy,
                    'classification_report': report,
                    'confusion_matrix': cm,
                    'fpr': fpr,
                    'tpr': tpr,
                    'roc_auc': roc_auc,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }
                
                # 显示评估结果
                self.eval_text.delete(1.0, tk.END)
                
                eval_text = f"模型评估结果\n"
                eval_text += f"{'='*50}\n\n"
                eval_text += f"模型准确率: {accuracy:.6f}\n\n"
                eval_text += f"ROC AUC分数: {roc_auc:.6f}\n\n"
                eval_text += f"分类报告:\n{report}\n\n"
                eval_text += f"混淆矩阵:\n{cm}\n\n"
                
                # 计算关键指标
                tn, fp, fn, tp = cm.ravel()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                eval_text += f"关键指标:\n"
                eval_text += f"• 精确率 (Precision): {precision:.6f}\n"
                eval_text += f"• 召回率 (Recall): {recall:.6f}\n"
                eval_text += f"• F1分数: {f1_score:.6f}\n"
                eval_text += f"• 真正例 (TP): {tp}\n"
                eval_text += f"• 假正例 (FP): {fp}\n"
                eval_text += f"• 假负例 (FN): {fn}\n"
                eval_text += f"• 真负例 (TN): {tn}\n"
                
                self.eval_text.insert(tk.END, eval_text)
                self.update_status("模型评估完成")
                
            except Exception as e:
                messagebox.showerror("错误", f"模型评估失败: {str(e)}")
                self.update_status("模型评估失败")
            finally:
                self.stop_progress()
        
        thread = threading.Thread(target=evaluate_model_thread)
        thread.daemon = True
        thread.start()
    
    def compare_models(self):
        """模型性能对比"""
        if self.X_train is None:
            messagebox.showerror("错误", "请先加载数据")
            return
            
        def compare_models_thread():
            try:
                self.start_progress()
                self.update_status("正在进行模型性能对比...")
                
                # 执行模型对比
                self.comparison_results = self.comparator.compare_models(
                    self.X_train, self.y_train, 
                    model_names=['XGBoost', 'RandomForest', 'LogisticRegression', 'DecisionTree', 'SVM']
                )
                
                # 显示对比结果
                self.compare_text.delete(1.0, tk.END)
                
                compare_text = f"模型性能对比结果\n"
                compare_text += f"{'='*60}\n\n"
                compare_text += str(self.comparison_results.to_string(index=False))
                
                # 获取最佳模型
                best_model_name, best_model, best_score = self.comparator.get_best_model('auc')
                if best_model_name:
                    compare_text += f"\n\n🏆 最佳模型: {best_model_name}"
                    compare_text += f"\n📊 最佳AUC分数: {best_score:.6f}"
                
                self.compare_text.insert(tk.END, compare_text)
                self.update_status("模型对比完成")
                
            except Exception as e:
                messagebox.showerror("错误", f"模型对比失败: {str(e)}")
                self.update_status("模型对比失败")
            finally:
                self.stop_progress()
        
        thread = threading.Thread(target=compare_models_thread)
        thread.daemon = True
        thread.start()
    
    def ablation_study(self):
        """消融实验"""
        if self.X_train is None:
            messagebox.showerror("错误", "请先加载数据")
            return
            
        def ablation_study_thread():
            try:
                self.start_progress()
                self.update_status("正在进行消融实验...")
                
                # 执行消融实验
                self.ablation_results = self.comparator.ablation_study(
                    self.X_train, self.y_train, base_model='XGBoost'
                )
                
                # 显示消融实验结果
                self.compare_text.delete(1.0, tk.END)
                
                ablation_text = f"消融实验结果\n"
                ablation_text += f"{'='*60}\n\n"
                ablation_text += str(self.ablation_results.to_string(index=False))
                
                self.compare_text.insert(tk.END, ablation_text)
                self.update_status("消融实验完成")
                
            except Exception as e:
                messagebox.showerror("错误", f"消融实验失败: {str(e)}")
                self.update_status("消融实验失败")
            finally:
                self.stop_progress()
        
        thread = threading.Thread(target=ablation_study_thread)
        thread.daemon = True
        thread.start()
    
    def save_model(self):
        """保存模型"""
        if self.model is None:
            messagebox.showerror("错误", "请先训练模型")
            return
            
        filename = filedialog.asksaveasfilename(
            title="保存模型",
            defaultextension=".pkl",
            filetypes=[("Pickle文件", "*.pkl"), ("所有文件", "*.*")]
        )
        
        if filename:
            try:
                self.trainer.save_model(self.model, filename)
                messagebox.showinfo("成功", f"模型已保存到: {filename}")
                self.update_status("模型保存完成")
            except Exception as e:
                messagebox.showerror("错误", f"模型保存失败: {str(e)}")
                self.update_status("模型保存失败")
    
    def plot_confusion_matrix(self):
        """绘制混淆矩阵"""
        if 'confusion_matrix' not in self.evaluation_results:
            messagebox.showerror("错误", "请先评估模型")
            return
            
        try:
            self.figure.clear()
            ax = self.figure.add_subplot(1, 1, 1)
            
            cm = self.evaluation_results['confusion_matrix']
            
            # 创建热力图
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       square=True, cbar_kws={'shrink': 0.8}, ax=ax)
            
            ax.set_title('混淆矩阵', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('预测标签', fontsize=12)
            ax.set_ylabel('真实标签', fontsize=12)
            
            # 设置刻度标签
            tick_labels = ['正常交易', '欺诈交易']
            ax.set_xticks([0.5, 1.5])
            ax.set_xticklabels(tick_labels)
            ax.set_yticks([0.5, 1.5])
            ax.set_yticklabels(tick_labels)
            
            self.canvas.draw()
            self.switch_to_plot_tab()  # 安全切换到图表标签页
            
        except Exception as e:
            messagebox.showerror("错误", f"绘制混淆矩阵失败: {str(e)}")
            
    def plot_roc_curve(self):
        """绘制ROC曲线"""
        if 'fpr' not in self.evaluation_results:
            messagebox.showerror("错误", "请先评估模型")
            return
            
        try:
            self.figure.clear()
            ax = self.figure.add_subplot(1, 1, 1)
            
            fpr = self.evaluation_results['fpr']
            tpr = self.evaluation_results['tpr']
            roc_auc = self.evaluation_results['roc_auc']
            
            # 绘制ROC曲线
            ax.plot(fpr, tpr, color='darkorange', lw=2, 
                   label=f'ROC曲线 (AUC = {roc_auc:.4f})')
            
            # 绘制对角线
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                   label='随机分类器 (AUC = 0.5)')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('假正例率 (False Positive Rate)', fontsize=12)
            ax.set_ylabel('真正例率 (True Positive Rate)', fontsize=12)
            ax.set_title('ROC曲线', fontsize=16, fontweight='bold', pad=20)
            ax.legend(loc="lower right", fontsize=12)
            ax.grid(True, alpha=0.3)
            
            self.canvas.draw()
            self.switch_to_plot_tab()  # 安全切换到图表标签页
            
        except Exception as e:
            messagebox.showerror("错误", f"绘制ROC曲线失败: {str(e)}")
            
    def plot_feature_importance(self):
        """绘制特征重要性"""
        if self.feature_importance_df is None:
            messagebox.showerror("错误", "请先评估模型")
            return
            
        try:
            self.figure.clear()
            ax = self.figure.add_subplot(1, 1, 1)
            
            # 选择前15个最重要的特征
            top_features = self.feature_importance_df.head(15)
            
            # 创建水平条形图
            bars = ax.barh(range(len(top_features)), top_features['Importance'], 
                          color='skyblue', edgecolor='navy', alpha=0.8)
            
            # 设置y轴标签
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['Feature'])
            ax.invert_yaxis()  # 反转y轴
            
            # 设置x轴标签
            ax.set_xlabel('重要性分数', fontsize=12)
            ax.set_title('特征重要性排名 (前15)', fontsize=16, fontweight='bold', pad=20)
            
            # 在条形图上添加数值标签
            for i, (bar, importance) in enumerate(zip(bars, top_features['Importance'])):
                ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                       f'{importance:.4f}', 
                       ha='left', va='center', fontsize=9)
            
            ax.grid(True, alpha=0.3, axis='x')
            
            self.canvas.draw()
            self.switch_to_plot_tab()  # 安全切换到图表标签页
            
        except Exception as e:
            messagebox.showerror("错误", f"绘制特征重要性失败: {str(e)}")
    
    def plot_comparison_results(self):
        """绘制模型对比结果图表"""
        if self.comparison_results is None:
            messagebox.showerror("错误", "请先进行模型对比")
            return
            
        try:
            self.figure.clear()
            
            # 提取F1分数用于绘图
            f1_scores = []
            model_names = []
            
            for idx, row in self.comparison_results.iterrows():
                f1_str = row['F1-Score']
                f1_score = float(f1_str.split(' ± ')[0])
                f1_scores.append(f1_score)
                model_names.append(row['Model'])
            
            # 创建子图
            ax1 = self.figure.add_subplot(1, 2, 1)
            ax2 = self.figure.add_subplot(1, 2, 2)
            
            # 子图1：F1分数对比
            bars1 = ax1.bar(model_names, f1_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
            ax1.set_title('模型F1分数对比', fontsize=14, fontweight='bold')
            ax1.set_ylabel('F1分数', fontsize=12)
            ax1.set_ylim(0, max(f1_scores) * 1.1)
            
            # 在柱状图上添加数值标签
            for bar, score in zip(bars1, f1_scores):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{score:.4f}', ha='center', va='bottom', fontsize=10)
            
            # 旋转x轴标签
            ax1.tick_params(axis='x', rotation=45)
            
            # 子图2：训练时间对比
            training_times = []
            for idx, row in self.comparison_results.iterrows():
                time_str = row['Training_Time']
                time_val = float(time_str.replace('s', ''))
                training_times.append(time_val)
            
            bars2 = ax2.bar(model_names, training_times, color=['#d62728', '#9467bd', '#8c564b'])
            ax2.set_title('模型训练时间对比', fontsize=14, fontweight='bold')
            ax2.set_ylabel('训练时间 (秒)', fontsize=12)
            
            # 在柱状图上添加数值标签
            for bar, time_val in zip(bars2, training_times):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{time_val:.3f}s', ha='center', va='bottom', fontsize=10)
            
            # 旋转x轴标签
            ax2.tick_params(axis='x', rotation=45)
            
            # 使用subplots_adjust替代tight_layout避免布局冲突
            self.figure.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15, wspace=0.3)
            self.canvas.draw()
            self.switch_to_plot_tab()  # 安全切换到图表标签页
            
        except Exception as e:
            messagebox.showerror("错误", f"绘制对比结果失败: {str(e)}")
    
    def save_current_plot(self):
        """保存当前显示的图表"""
        try:
            # 选择保存路径
            file_path = filedialog.asksaveasfilename(
                title="保存图表",
                defaultextension=".png",
                filetypes=[
                    ("PNG图片", "*.png"),
                    ("JPG图片", "*.jpg"),
                    ("PDF文件", "*.pdf"),
                    ("SVG文件", "*.svg"),
                    ("所有文件", "*.*")
                ]
            )
            
            if file_path:
                # 保存图表
                self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
                messagebox.showinfo("成功", f"图表已保存到: {file_path}")
                self.update_status(f"图表已保存: {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror("错误", f"保存图表失败: {str(e)}")
    
    def save_all_plots(self):
        """保存所有图表到指定目录"""
        try:
            if self.model is None:
                messagebox.showerror("错误", "请先训练模型")
                return
            
            # 选择保存目录
            save_dir = filedialog.askdirectory(title="选择保存目录")
            
            if save_dir:
                self.start_progress()
                self.update_status("正在保存所有图表...")
                
                def save_plots_thread():
                    try:
                        import datetime
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        # 1. 混淆矩阵
                        if self.evaluation_results:
                            from utils import plot_confusion_matrix
                            cm_file = os.path.join(save_dir, f"confusion_matrix_{timestamp}.png")
                            
                            # 重新计算混淆矩阵
                            from sklearn.metrics import confusion_matrix
                            y_pred = self.model.predict(self.X_test_enhanced)
                            cm = confusion_matrix(self.y_test_enhanced, y_pred)
                            
                            plt.figure(figsize=(8, 6))
                            import seaborn as sns
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                       annot_kws={'size': 12}, cbar=False)
                            plt.title('混淆矩阵', fontsize=14, fontweight='bold')
                            plt.xlabel('预测标签', fontsize=12)
                            plt.ylabel('真实标签', fontsize=12)
                            plt.tight_layout()
                            plt.savefig(cm_file, dpi=300, bbox_inches='tight')
                            plt.close()
                        
                        # 2. ROC曲线
                        if self.evaluation_results and 'roc_auc' in self.evaluation_results:
                            from utils import plot_roc_curve
                            from sklearn.metrics import roc_curve
                            
                            y_pred_proba = self.model.predict_proba(self.X_test_enhanced)[:, 1]
                            fpr, tpr, _ = roc_curve(self.y_test_enhanced, y_pred_proba)
                            roc_auc = self.evaluation_results['roc_auc']
                            
                            roc_file = os.path.join(save_dir, f"roc_curve_{timestamp}.png")
                            plt.figure(figsize=(8, 6))
                            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                                   label=f'ROC曲线 (AUC = {roc_auc:.4f})')
                            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                                   label='随机分类器 (AUC = 0.5)')
                            plt.xlim([0.0, 1.0])
                            plt.ylim([0.0, 1.05])
                            plt.xlabel('假正例率 (FPR)', fontsize=12)
                            plt.ylabel('真正例率 (TPR)', fontsize=12)
                            plt.title('ROC曲线', fontsize=14, fontweight='bold')
                            plt.legend(loc="lower right")
                            plt.grid(True, alpha=0.3)
                            plt.tight_layout()
                            plt.savefig(roc_file, dpi=300, bbox_inches='tight')
                            plt.close()
                        
                        # 3. 特征重要性
                        if self.feature_importance_df is not None:
                            feature_file = os.path.join(save_dir, f"feature_importance_{timestamp}.png")
                            
                            plt.figure(figsize=(10, 8))
                            top_features = self.feature_importance_df.head(10)
                            bars = plt.barh(range(len(top_features)), top_features['Importance'], 
                                          color='skyblue', alpha=0.8)
                            
                            plt.yticks(range(len(top_features)), top_features['Feature'])
                            plt.xlabel('重要性分数', fontsize=12)
                            plt.title('Top 10 特征重要性', fontsize=14, fontweight='bold')
                            plt.grid(True, alpha=0.3, axis='x')
                            
                            # 添加数值标签
                            for i, bar in enumerate(bars):
                                width = bar.get_width()
                                plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                                       f'{width:.3f}', ha='left', va='center', fontsize=10)
                            
                            plt.tight_layout()
                            plt.savefig(feature_file, dpi=300, bbox_inches='tight')
                            plt.close()
                        
                        # 4. 模型对比结果
                        if self.comparison_results is not None:
                            comparison_file = os.path.join(save_dir, f"model_comparison_{timestamp}.png")
                            
                            plt.figure(figsize=(12, 8))
                            
                            # 性能指标对比
                            metrics = ['AUC', 'F1-Score', 'Precision', 'Recall']
                            models = self.comparison_results.index.tolist()
                            
                            x = np.arange(len(models))
                            width = 0.2
                            
                            for i, metric in enumerate(metrics):
                                if metric in self.comparison_results.columns:
                                    values = self.comparison_results[metric]
                                    plt.bar(x + i*width, values, width, label=metric, alpha=0.8)
                            
                            plt.xlabel('模型', fontsize=12)
                            plt.ylabel('分数', fontsize=12)
                            plt.title('模型性能对比', fontsize=14, fontweight='bold')
                            plt.xticks(x + width*1.5, models, rotation=45)
                            plt.legend()
                            plt.grid(True, alpha=0.3)
                            plt.tight_layout()
                            plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
                            plt.close()
                        
                        # 5. 生成报告文件
                        report_file = os.path.join(save_dir, f"experiment_report_{timestamp}.txt")
                        with open(report_file, 'w', encoding='utf-8') as f:
                            f.write("信用卡欺诈检测实验报告\n")
                            f.write("="*50 + "\n\n")
                            f.write(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                            
                            if self.evaluation_results:
                                f.write("模型性能:\n")
                                f.write("-"*30 + "\n")
                                for key, value in self.evaluation_results.items():
                                    if isinstance(value, float):
                                        f.write(f"{key}: {value:.6f}\n")
                                    else:
                                        f.write(f"{key}: {value}\n")
                            
                            if self.comparison_results is not None:
                                f.write("\n模型对比结果:\n")
                                f.write("-"*30 + "\n")
                                f.write(self.comparison_results.to_string())
                        
                        self.stop_progress()
                        messagebox.showinfo("成功", f"所有图表已保存到: {save_dir}")
                        self.update_status(f"图表保存完成: {len(os.listdir(save_dir))} 个文件")
                        
                    except Exception as e:
                        self.stop_progress()
                        messagebox.showerror("错误", f"保存图表失败: {str(e)}")
                
                # 在新线程中执行保存操作
                import threading
                thread = threading.Thread(target=save_plots_thread)
                thread.daemon = True
                thread.start()
            
        except Exception as e:
            messagebox.showerror("错误", f"保存图表失败: {str(e)}")
    
    def show_visualization_menu(self):
        """显示可视化分析菜单"""
        if self.model is None:
            messagebox.showerror("错误", "请先训练模型")
            return
        
        # 创建可视化选择窗口
        viz_window = tk.Toplevel(self.root)
        viz_window.title("可视化分析")
        viz_window.geometry("300x200")
        viz_window.resizable(False, False)
        
        # 居中显示
        viz_window.transient(self.root)
        viz_window.grab_set()
        
        ttk.Label(viz_window, text="选择可视化类型:", font=('Arial', 12, 'bold')).pack(pady=20)
        
        ttk.Button(viz_window, text="ROC曲线", 
                 command=lambda: [viz_window.destroy(), self.plot_roc_curve()]).pack(pady=5, padx=20, fill=tk.X)
        ttk.Button(viz_window, text="混淆矩阵", 
                 command=lambda: [viz_window.destroy(), self.plot_confusion_matrix()]).pack(pady=5, padx=20, fill=tk.X)
        ttk.Button(viz_window, text="PR曲线", 
                 command=lambda: [viz_window.destroy(), self.plot_pr_curve()]).pack(pady=5, padx=20, fill=tk.X)
        ttk.Button(viz_window, text="取消", command=viz_window.destroy).pack(pady=10)
    
    def plot_pr_curve(self):
        """绘制PR曲线"""
        try:
            if self.model is None:
                messagebox.showerror("错误", "请先训练模型")
                return
            
            self.start_progress()
            self.update_status("正在绘制PR曲线...")
            
            def plot_pr_thread():
                try:
                    from sklearn.metrics import precision_recall_curve, average_precision_score
                    from utils import plot_precision_recall_curve
                    
                    y_pred_proba = self.model.predict_proba(self.X_test_enhanced)[:, 1]
                    precision, recall, _ = precision_recall_curve(self.y_test_enhanced, y_pred_proba)
                    average_precision = average_precision_score(self.y_test_enhanced, y_pred_proba)
                    
                    self.figure.clear()
                    plot_precision_recall_curve(precision, recall, average_precision, 
                                            title="精确率-召回率曲线", figsize=(12, 8))
                    self.canvas.draw()
                    self.switch_to_plot_tab()
                    
                    self.stop_progress()
                    self.update_status("PR曲线绘制完成")
                    
                except Exception as e:
                    self.stop_progress()
                    messagebox.showerror("错误", f"绘制PR曲线失败: {str(e)}")
            
            thread = threading.Thread(target=plot_pr_thread)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            messagebox.showerror("错误", f"绘制PR曲线失败: {str(e)}")
    
    def export_results(self):
        """导出实验结果"""
        try:
            if self.model is None:
                messagebox.showerror("错误", "请先训练模型")
                return
            
            # 选择保存目录
            save_dir = filedialog.askdirectory(title="选择导出目录")
            
            if save_dir:
                self.start_progress()
                self.update_status("正在导出结果...")
                
                def export_thread():
                    try:
                        import datetime
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        # 1. 导出评估结果
                        if self.evaluation_results:
                            eval_file = os.path.join(save_dir, f"evaluation_results_{timestamp}.txt")
                            with open(eval_file, 'w', encoding='utf-8') as f:
                                f.write("模型评估结果\n")
                                f.write("="*50 + "\n\n")
                                f.write(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                                for key, value in self.evaluation_results.items():
                                    if isinstance(value, float):
                                        f.write(f"{key}: {value:.6f}\n")
                                    else:
                                        f.write(f"{key}: {value}\n")
                        
                        # 2. 导出特征重要性
                        if self.feature_importance_df is not None:
                            feature_file = os.path.join(save_dir, f"feature_importance_{timestamp}.csv")
                            self.feature_importance_df.to_csv(feature_file, index=False, encoding='utf-8')
                        
                        # 3. 导出模型对比结果
                        if self.comparison_results is not None:
                            comparison_file = os.path.join(save_dir, f"model_comparison_{timestamp}.csv")
                            self.comparison_results.to_csv(comparison_file, index=True, encoding='utf-8')
                        
                        # 4. 保存主要图表
                        if self.evaluation_results:
                            # ROC曲线
                            roc_file = os.path.join(save_dir, f"roc_curve_{timestamp}.png")
                            y_pred_proba = self.model.predict_proba(self.X_test_enhanced)[:, 1]
                            from sklearn.metrics import roc_curve
                            fpr, tpr, _ = roc_curve(self.y_test_enhanced, y_pred_proba)
                            roc_auc = self.evaluation_results.get('roc_auc', 0)
                            
                            plt.figure(figsize=(8, 6))
                            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                                   label=f'ROC曲线 (AUC = {roc_auc:.4f})')
                            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                            plt.xlim([0.0, 1.0])
                            plt.ylim([0.0, 1.05])
                            plt.xlabel('假正例率')
                            plt.ylabel('真正例率')
                            plt.title('ROC曲线')
                            plt.legend(loc="lower right")
                            plt.grid(True, alpha=0.3)
                            plt.savefig(roc_file, dpi=300, bbox_inches='tight')
                            plt.close()
                        
                        self.stop_progress()
                        messagebox.showinfo("成功", f"结果已导出到: {save_dir}")
                        self.update_status(f"导出完成: {len(os.listdir(save_dir))} 个文件")
                        
                    except Exception as e:
                        self.stop_progress()
                        messagebox.showerror("错误", f"导出失败: {str(e)}")
                
                thread = threading.Thread(target=export_thread)
                thread.daemon = True
                thread.start()
            
        except Exception as e:
            messagebox.showerror("错误", f"导出失败: {str(e)}")

def main():
    """主函数"""
    root = tk.Tk()
    app = EnhancedCreditCardFraudGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
