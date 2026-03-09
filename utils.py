#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具函数模块
包含绘图、数据处理等辅助功能

作者: 转换自Jupyter Notebook
日期: 2026年3月9日
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.style.use('seaborn-v0_8')

def plot_confusion_matrix(cm, title="混淆矩阵", figsize=(8, 6)):
    """
    绘制混淆矩阵热力图
    
    参数:
        cm (array): 混淆矩阵
        title (str): 图表标题
        figsize (tuple): 图表大小
    """
    plt.figure(figsize=figsize)
    
    # 创建热力图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                square=True, cbar_kws={'shrink': 0.8})
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    
    # 设置刻度标签
    tick_labels = ['正常交易', '欺诈交易']
    plt.xticks([0.5, 1.5], tick_labels)
    plt.yticks([0.5, 1.5], tick_labels)
    
    # 添加网格线
    plt.grid(False)
    
    plt.tight_layout()
    plt.show()

def plot_roc_curve(fpr, tpr, roc_auc, title="ROC曲线", figsize=(8, 6)):
    """
    绘制ROC曲线
    
    参数:
        fpr (array): 假正例率
        tpr (array): 真正例率
        roc_auc (float): AUC值
        title (str): 图表标题
        figsize (tuple): 图表大小
    """
    plt.figure(figsize=figsize)
    
    # 绘制ROC曲线
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC曲线 (AUC = {roc_auc:.4f})')
    
    # 绘制对角线（随机分类器）
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='随机分类器 (AUC = 0.5)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正例率 (False Positive Rate)', fontsize=12)
    plt.ylabel('真正例率 (True Positive Rate)', fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc="lower right", fontsize=12)
    
    # 添加网格
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_feature_importance(feature_importance_df, title="特征重要性", top_n=10, figsize=(10, 8)):
    """
    绘制特征重要性条形图
    
    参数:
        feature_importance_df (pd.DataFrame): 特征重要性数据框
        title (str): 图表标题
        top_n (int): 显示前N个重要特征
        figsize (tuple): 图表大小
    """
    # 选择前N个最重要的特征
    top_features = feature_importance_df.head(top_n)
    
    plt.figure(figsize=figsize)
    
    # 创建水平条形图
    bars = plt.barh(range(len(top_features)), top_features['Importance'], 
                    color='skyblue', edgecolor='navy', alpha=0.8)
    
    # 设置y轴标签
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.gca().invert_yaxis()  # 反转y轴，使重要性最高的在顶部
    
    # 设置x轴标签
    plt.xlabel('重要性分数', fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    
    # 在条形图上添加数值标签
    for i, (bar, importance) in enumerate(zip(bars, top_features['Importance'])):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{importance:.4f}', 
                ha='left', va='center', fontsize=10)
    
    # 添加网格线
    plt.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()

def plot_class_distribution(y, title="类别分布", figsize=(8, 6)):
    """
    绘制类别分布图
    
    参数:
        y (array): 目标变量
        title (str): 图表标题
        figsize (tuple): 图表大小
    """
    plt.figure(figsize=figsize)
    
    # 计算类别数量和百分比
    class_counts = pd.Series(y).value_counts().sort_index()
    class_percentages = class_counts / class_counts.sum() * 100
    
    # 创建条形图
    bars = plt.bar(class_counts.index, class_counts.values, 
                   color=['lightcoral', 'lightblue'], alpha=0.8, edgecolor='black')
    
    # 设置标题和标签
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('类别', fontsize=12)
    plt.ylabel('样本数量', fontsize=12)
    
    # 设置x轴刻度
    plt.xticks(class_counts.index, ['正常交易 (0)', '欺诈交易 (1)'])
    
    # 在条形图上添加数值和百分比标签
    for i, (bar, count, percentage) in enumerate(zip(bars, class_counts.values, class_percentages.values)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + class_counts.max()*0.01,
                f'{count}\n({percentage:.2f}%)', 
                ha='center', va='bottom', fontsize=11)
    
    # 添加网格线
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()

def plot_precision_recall_curve(precision, recall, average_precision, title="精确率-召回率曲线", figsize=(8, 6)):
    """
    绘制精确率-召回率曲线
    
    参数:
        precision (array): 精确率
        recall (array): 召回率
        average_precision (float): 平均精确率
        title (str): 图表标题
        figsize (tuple): 图表大小
    """
    plt.figure(figsize=figsize)
    
    # 绘制PR曲线
    plt.plot(recall, precision, color='darkblue', lw=2, 
             label=f'PR曲线 (AP = {average_precision:.4f})')
    
    # 绘制基线
    baseline = len(precision[precision == 1]) / len(precision)
    plt.axhline(y=baseline, color='red', linestyle='--', 
                label=f'基线 (AP = {baseline:.4f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('召回率 (Recall)', fontsize=12)
    plt.ylabel('精确率 (Precision)', fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc="upper right", fontsize=12)
    
    # 添加网格
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_learning_curves(train_scores, val_scores, train_sizes, title="学习曲线", figsize=(10, 6)):
    """
    绘制学习曲线
    
    参数:
        train_scores (array): 训练集得分
        val_scores (array): 验证集得分
        train_sizes (array): 训练样本数量
        title (str): 图表标题
        figsize (tuple): 图表大小
    """
    plt.figure(figsize=figsize)
    
    # 计算均值和标准差
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # 绘制训练集曲线
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='训练集')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.1, color='blue')
    
    # 绘制验证集曲线
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='验证集')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                     alpha=0.1, color='red')
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('训练样本数量', fontsize=12)
    plt.ylabel('得分', fontsize=12)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def create_summary_report(evaluation_results, feature_importance_df=None):
    """
    创建评估报告摘要
    
    参数:
        evaluation_results (dict): 评估结果
        feature_importance_df (pd.DataFrame): 特征重要性数据框
        
    返回:
        str: 格式化的报告字符串
    """
    report = []
    report.append("=" * 60)
    report.append("信用卡欺诈检测模型评估报告")
    report.append("=" * 60)
    
    # 基础指标
    report.append("\n📊 基础性能指标:")
    report.append(f"• 准确率: {evaluation_results.get('accuracy', 0):.6f}")
    report.append(f"• 精确率: {evaluation_results.get('precision', 0):.6f}")
    report.append(f"• 召回率: {evaluation_results.get('recall', 0):.6f}")
    report.append(f"• F1分数: {evaluation_results.get('f1_score', 0):.6f}")
    
    # 混淆矩阵分析
    if 'true_positives' in evaluation_results:
        report.append("\n🔍 混淆矩阵分析:")
        report.append(f"• 真正例 (TP): {evaluation_results['true_positives']}")
        report.append(f"• 假正例 (FP): {evaluation_results['false_positives']}")
        report.append(f"• 假负例 (FN): {evaluation_results['false_negatives']}")
        report.append(f"• 真负例 (TN): {evaluation_results['true_negatives']}")
        report.append(f"• 特异性: {evaluation_results.get('specificity', 0):.6f}")
        report.append(f"• 敏感性: {evaluation_results.get('sensitivity', 0):.6f}")
    
    # ROC AUC
    if 'roc_auc' in evaluation_results:
        report.append(f"\n📈 ROC AUC: {evaluation_results['roc_auc']:.6f}")
    
    # 特征重要性
    if feature_importance_df is not None:
        report.append("\n🏆 前5个最重要特征:")
        top_5 = feature_importance_df.head(5)
        for _, row in top_5.iterrows():
            report.append(f"• {row['Feature']}: {row['Importance']:.6f}")
    
    report.append("\n" + "=" * 60)
    
    return "\n".join(report)

def save_results_to_csv(results, filename="model_results.csv"):
    """
    将评估结果保存到CSV文件
    
    参数:
        results (dict): 评估结果
        filename (str): 文件名
    """
    # 准备数据
    data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Specificity'],
        'Value': [
            results.get('accuracy', 0),
            results.get('precision', 0),
            results.get('recall', 0),
            results.get('f1_score', 0),
            results.get('roc_auc', 0),
            results.get('specificity', 0)
        ]
    }
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"结果已保存到 {filename}")

# 测试函数
def test_utils():
    """测试工具函数"""
    print("测试工具函数...")
    
    # 创建模拟数据
    cm = np.array([[100, 5], [10, 15]])
    fpr = np.array([0, 0.1, 0.2, 0.3, 1])
    tpr = np.array([0, 0.7, 0.8, 0.9, 1])
    roc_auc = 0.85
    
    feature_importance = pd.DataFrame({
        'Feature': ['V17', 'V12', 'V14', 'V4', 'V10'],
        'Importance': [0.6789, 0.1306, 0.0221, 0.0165, 0.0120]
    })
    
    evaluation_results = {
        'accuracy': 0.9995,
        'precision': 0.89,
        'recall': 0.83,
        'f1_score': 0.86,
        'roc_auc': 0.9786,
        'true_positives': 81,
        'false_positives': 10,
        'false_negatives': 17,
        'true_negatives': 56854,
        'specificity': 0.9998,
        'sensitivity': 0.8265
    }
    
    # 测试绘图函数（注释掉以避免显示）
    # plot_confusion_matrix(cm)
    # plot_roc_curve(fpr, tpr, roc_auc)
    # plot_feature_importance(feature_importance)
    
    # 测试报告生成
    report = create_summary_report(evaluation_results, feature_importance)
    print("✓ 工具函数测试完成")
    print("\n示例报告:")
    print(report)

if __name__ == "__main__":
    test_utils()
