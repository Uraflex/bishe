#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    def __init__(self):
        self.evaluation_results = {}
        
    def calculate_accuracy(self, y_true, y_pred):
        """
        计算准确率
        
        参数:
            y_true (array): 真实标签
            y_pred (array): 预测标签
            
        返回:
            float: 准确率
        """
        return accuracy_score(y_true, y_pred)
    
    def calculate_precision(self, y_true, y_pred, pos_label=1):
        """
        计算精确率
        
        参数:
            y_true (array): 真实标签
            y_pred (array): 预测标签
            pos_label (int): 正类标签
            
        返回:
            float: 精确率
        """
        return precision_score(y_true, y_pred, pos_label=pos_label)
    
    def calculate_recall(self, y_true, y_pred, pos_label=1):
        """
        计算召回率
        
        参数:
            y_true (array): 真实标签
            y_pred (array): 预测标签
            pos_label (int): 正类标签
            
        返回:
            float: 召回率
        """
        return recall_score(y_true, y_pred, pos_label=pos_label)
    
    def calculate_f1_score(self, y_true, y_pred, pos_label=1):
        """
        计算F1分数
        
        参数:
            y_true (array): 真实标签
            y_pred (array): 预测标签
            pos_label (int): 正类标签
            
        返回:
            float: F1分数
        """
        return f1_score(y_true, y_pred, pos_label=pos_label)
    
    def classification_report(self, y_true, y_pred, target_names=None):
        """
        生成分类报告
        
        参数:
            y_true (array): 真实标签
            y_pred (array): 预测标签
            target_names (list): 类别名称
            
        返回:
            str: 分类报告
        """
        if target_names is None:
            target_names = ['正常交易', '欺诈交易']
            
        return classification_report(y_true, y_pred, target_names=target_names)
    
    def confusion_matrix(self, y_true, y_pred):
        """
        计算混淆矩阵
        
        参数:
            y_true (array): 真实标签
            y_pred (array): 预测标签
            
        返回:
            array: 混淆矩阵
        """
        return confusion_matrix(y_true, y_pred)
    
    def calculate_roc_auc(self, y_true, y_pred_proba):
        """
        计算ROC曲线和AUC分数
        
        参数:
            y_true (array): 真实标签
            y_pred_proba (array): 预测概率
            
        返回:
            tuple: (fpr, tpr, roc_auc)
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, roc_auc
    
    def calculate_precision_recall_curve(self, y_true, y_pred_proba):
        """
        计算精确率-召回率曲线
        
        参数:
            y_true (array): 真实标签
            y_pred_proba (array): 预测概率
            
        返回:
            tuple: (precision, recall, average_precision)
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        average_precision = average_precision_score(y_true, y_pred_proba)
        return precision, recall, average_precision
    
    def get_feature_importance(self, model, feature_names):
        """
        获取特征重要性
        
        参数:
            model: 训练好的模型
            feature_names (list): 特征名称列表
            
        返回:
            pd.DataFrame: 特征重要性数据框
        """
        if hasattr(model, 'feature_importances_'):
            importance_scores = model.feature_importances_
        elif hasattr(model, 'get_booster'):
            # XGBoost模型
            importance_scores = model.get_booster().get_score(importance_type='weight')
            # 转换为与特征名称对应的数组
            importance_scores = [importance_scores.get(f'f{i}', 0) for i in range(len(feature_names))]
        else:
            print("警告: 模型不支持特征重要性分析")
            return None
        
        # 创建特征重要性数据框
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_scores
        })
        
        # 按重要性排序
        feature_importance_df = feature_importance_df.sort_values(
            by='Importance', ascending=False
        ).reset_index(drop=True)
        
        return feature_importance_df
    
    def comprehensive_evaluation(self, y_true, y_pred, y_pred_proba=None):
        """
        综合评估模型性能
        
        参数:
            y_true (array): 真实标签
            y_pred (array): 预测标签
            y_pred_proba (array): 预测概率（可选）
            
        返回:
            dict: 评估结果字典
        """
        results = {}
        
        # 基础指标
        results['accuracy'] = self.calculate_accuracy(y_true, y_pred)
        results['precision'] = self.calculate_precision(y_true, y_pred)
        results['recall'] = self.calculate_recall(y_true, y_pred)
        results['f1_score'] = self.calculate_f1_score(y_true, y_pred)
        
        # 混淆矩阵
        cm = self.confusion_matrix(y_true, y_pred)
        results['confusion_matrix'] = cm
        
        # 解析混淆矩阵
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            results['true_negatives'] = tn
            results['false_positives'] = fp
            results['false_negatives'] = fn
            results['true_positives'] = tp
            
            # 计算额外指标
            results['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            results['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            results['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
            results['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # ROC AUC（如果提供了预测概率）
        if y_pred_proba is not None:
            fpr, tpr, roc_auc = self.calculate_roc_auc(y_true, y_pred_proba)
            results['roc_auc'] = roc_auc
            results['fpr'] = fpr
            results['tpr'] = tpr
            
            # 精确率-召回率曲线
            precision, recall, avg_precision = self.calculate_precision_recall_curve(
                y_true, y_pred_proba
            )
            results['average_precision'] = avg_precision
            results['precision_curve'] = precision
            results['recall_curve'] = recall
        
        # 保存结果
        self.evaluation_results = results
        
        return results
    
    def print_evaluation_summary(self, results=None):
        """
        打印评估结果摘要
        
        参数:
            results (dict): 评估结果字典
        """
        if results is None:
            results = self.evaluation_results
            
        if not results:
            print("没有可显示的评估结果")
            return
        
        print("\n" + "="*50)
        print("模型性能评估摘要")
        print("="*50)
        
        # 基础指标
        print(f"准确率 (Accuracy): {results.get('accuracy', 0):.6f}")
        print(f"精确率 (Precision): {results.get('precision', 0):.6f}")
        print(f"召回率 (Recall): {results.get('recall', 0):.6f}")
        print(f"F1分数 (F1-Score): {results.get('f1_score', 0):.6f}")
        
        # 混淆矩阵相关指标
        if 'true_positives' in results:
            print(f"\n混淆矩阵分析:")
            print(f"真正例 (TP): {results['true_positives']}")
            print(f"假正例 (FP): {results['false_positives']}")
            print(f"假负例 (FN): {results['false_negatives']}")
            print(f"真负例 (TN): {results['true_negatives']}")
            print(f"特异性 (Specificity): {results.get('specificity', 0):.6f}")
            print(f"敏感性 (Sensitivity): {results.get('sensitivity', 0):.6f}")
        
        # ROC AUC
        if 'roc_auc' in results:
            print(f"\nROC AUC: {results['roc_auc']:.6f}")
        
        # 平均精确率
        if 'average_precision' in results:
            print(f"平均精确率 (AP): {results['average_precision']:.6f}")
        
        print("="*50)
    
    def compare_models(self, model_results_dict):
        """
        比较多个模型的性能
        
        参数:
            model_results_dict (dict): 模型结果字典
                {模型名称: 评估结果字典}
                
        返回:
            pd.DataFrame: 模型比较表
        """
        comparison_data = []
        
        for model_name, results in model_results_dict.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': results.get('accuracy', 0),
                'Precision': results.get('precision', 0),
                'Recall': results.get('recall', 0),
                'F1-Score': results.get('f1_score', 0),
                'ROC-AUC': results.get('roc_auc', 0),
                'Specificity': results.get('specificity', 0)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.round(6)
        
        return comparison_df
