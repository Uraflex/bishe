#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信用卡欺诈检测系统 - 主程序
基于XGBoost的信用卡交易反欺诈模型

作者: 转换自Jupyter Notebook
日期: 2026年3月9日
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report, 
                           confusion_matrix, roc_curve, auc)
import warnings
warnings.filterwarnings('ignore')

# 导入自定义模块
from data_processor import DataProcessor
from model_trainer import ModelTrainer
from evaluator import ModelEvaluator
from utils import plot_confusion_matrix, plot_roc_curve, plot_feature_importance

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def main():
    """主函数：执行完整的信用卡欺诈检测流程"""
    
    print("=" * 60)
    print("信用卡欺诈检测系统启动")
    print("基于XGBoost的机器学习模型")
    print("=" * 60)
    
    # 1. 数据加载和预处理
    print("\n1. 数据加载和预处理阶段")
    print("-" * 40)
    
    # 初始化数据处理器
    processor = DataProcessor()
    
    # 加载数据集
    df = processor.load_data(r"d:\我的文档\Desktop\11\creditcard.csv")
    print(f"数据集形状: {df.shape}")
    
    # 数据探索
    processor.explore_data(df)
    
    # 数据预处理
    X, y = processor.preprocess_data(df)
    print(f"特征矩阵形状: {X.shape}")
    print(f"目标变量形状: {y.shape}")
    
    # 2. 数据分割
    print("\n2. 数据分割阶段")
    print("-" * 40)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 3. 模型训练
    print("\n3. 模型训练阶段")
    print("-" * 40)
    
    # 初始化模型训练器
    trainer = ModelTrainer()
    
    # 训练XGBoost模型
    model = trainer.train_xgboost(X_train, y_train)
    print("XGBoost模型训练完成")
    
    # 4. 模型预测
    print("\n4. 模型预测阶段")
    print("-" * 40)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    print("预测完成")
    
    # 5. 模型评估
    print("\n5. 模型评估阶段")
    print("-" * 40)
    
    # 初始化评估器
    evaluator = ModelEvaluator()
    
    # 基础评估指标
    accuracy = evaluator.calculate_accuracy(y_test, y_pred)
    report = evaluator.classification_report(y_test, y_pred)
    
    print(f"模型准确率: {accuracy:.6f}")
    print("\n分类报告:")
    print(report)
    
    # 混淆矩阵
    cm = evaluator.confusion_matrix(y_test, y_pred)
    print("\n混淆矩阵:")
    print(cm)
    
    # ROC曲线和AUC分数
    fpr, tpr, roc_auc = evaluator.calculate_roc_auc(y_test, y_pred_proba)
    print(f"\nROC AUC分数: {roc_auc:.6f}")
    
    # 6. 可视化结果
    print("\n6. 结果可视化")
    print("-" * 40)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(cm, "混淆矩阵")
    
    # 绘制ROC曲线
    plot_roc_curve(fpr, tpr, roc_auc, "ROC曲线")
    
    # 特征重要性分析
    feature_names = X.columns.tolist()
    feature_importance = evaluator.get_feature_importance(model, feature_names)
    
    # 绘制特征重要性
    plot_feature_importance(feature_importance, "特征重要性排名")
    
    # 7. 输出重要结果
    print("\n7. 重要结果总结")
    print("-" * 40)
    
    print(f"✓ 模型准确率: {accuracy:.6f}")
    print(f"✓ ROC AUC分数: {roc_auc:.6f}")
    
    # 计算欺诈检测的关键指标
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"✓ 精确率 (Precision): {precision:.6f}")
    print(f"✓ 召回率 (Recall): {recall:.6f}")
    print(f"✓ F1分数: {f1_score:.6f}")
    print(f"✓ 真正例 (TP): {tp}")
    print(f"✓ 假正例 (FP): {fp}")
    print(f"✓ 假负例 (FN): {fn}")
    print(f"✓ 真负例 (TN): {tn}")
    
    # 8. 保存模型
    print("\n8. 保存模型")
    print("-" * 40)
    
    trainer.save_model(model, "xgboost_fraud_detection_model.pkl")
    print("模型已保存为 'xgboost_fraud_detection_model.pkl'")
    
    print("\n" + "=" * 60)
    print("信用卡欺诈检测系统运行完成")
    print("=" * 60)

if __name__ == "__main__":
    main()
