#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型性能对比模块
实现多种算法的性能对比和消融实验

作者: 基于开题报告要求
日期: 2026年3月9日
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, classification_report, 
                           confusion_matrix, roc_curve)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import xgboost as xgb
import time
import warnings
warnings.filterwarnings('ignore')

class ModelComparator:
    """模型性能对比器"""
    
    def __init__(self, cv_folds=5, random_state=42):
        """
        初始化模型对比器
        
        参数:
            cv_folds (int): 交叉验证折数
            random_state (int): 随机种子
        """
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.comparison_results = {}
        
        # 定义要对比的模型
        self.model_definitions = {
            'XGBoost': {
                'model': xgb.XGBClassifier,
                'params': {
                    'random_state': random_state,
                    'n_jobs': -1,
                    'eval_metric': 'logloss'
                },
                'param_grid': {
                    'n_estimators': [100, 200],
                    'max_depth': [4, 6],
                    'learning_rate': [0.1, 0.05]
                }
            },
            'RandomForest': {
                'model': RandomForestClassifier,
                'params': {
                    'random_state': random_state,
                    'n_jobs': -1
                },
                'param_grid': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20],
                    'min_samples_split': [2, 5]
                }
            },
            'LogisticRegression': {
                'model': LogisticRegression,
                'params': {
                    'random_state': random_state,
                    'max_iter': 1000
                },
                'param_grid': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l2']
                }
            },
            'DecisionTree': {
                'model': DecisionTreeClassifier,
                'params': {
                    'random_state': random_state
                },
                'param_grid': {
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'SVM': {
                'model': SVC,
                'params': {
                    'random_state': random_state,
                    'probability': True
                },
                'param_grid': {
                    'C': [0.1, 1.0],
                    'kernel': ['rbf']
                }
            }
        }
    
    def train_single_model(self, model_name, X, y, params=None):
        """
        训练单个模型
        
        参数:
            model_name (str): 模型名称
            X (pd.DataFrame): 特征矩阵
            y (pd.Series): 目标变量
            params (dict): 额外参数
            
        返回:
            dict: 训练结果
        """
        if model_name not in self.model_definitions:
            raise ValueError(f"不支持的模型: {model_name}")
        
        model_def = self.model_definitions[model_name]
        model_params = model_def['params'].copy()
        
        if params:
            model_params.update(params)
        
        # 创建模型
        model = model_def['model'](**model_params)
        
        # 训练时间
        start_time = time.time()
        
        # 交叉验证
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        # 计算各项指标
        accuracy_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        precision_scores = cross_val_score(model, X, y, cv=cv, scoring='precision')
        recall_scores = cross_val_score(model, X, y, cv=cv, scoring='recall')
        f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
        
        # ROC AUC需要特殊处理
        try:
            auc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
        except:
            auc_scores = [0] * self.cv_folds
        
        training_time = time.time() - start_time
        
        # 训练最终模型
        model.fit(X, y)
        
        results = {
            'model': model,
            'accuracy_mean': np.mean(accuracy_scores),
            'accuracy_std': np.std(accuracy_scores),
            'precision_mean': np.mean(precision_scores),
            'precision_std': np.std(precision_scores),
            'recall_mean': np.mean(recall_scores),
            'recall_std': np.std(recall_scores),
            'f1_mean': np.mean(f1_scores),
            'f1_std': np.std(f1_scores),
            'auc_mean': np.mean(auc_scores),
            'auc_std': np.std(auc_scores),
            'training_time': training_time,
            'params': model_params
        }
        
        return results
    
    def compare_models(self, X, y, model_names=None):
        """
        对比多个模型的性能
        
        参数:
            X (pd.DataFrame): 特征矩阵
            y (pd.Series): 目标变量
            model_names (list): 要对比的模型名称列表，None表示全部
            
        返回:
            pd.DataFrame: 对比结果
        """
        if model_names is None:
            model_names = list(self.model_definitions.keys())
        
        print(f"开始模型性能对比，对比模型: {model_names}")
        print("=" * 60)
        
        comparison_data = []
        
        for model_name in model_names:
            print(f"\n训练 {model_name} 模型...")
            print("-" * 40)
            
            try:
                results = self.train_single_model(model_name, X, y)
                self.results[model_name] = results
                
                # 记录对比数据
                comparison_row = {
                    'Model': model_name,
                    'Accuracy': f"{results['accuracy_mean']:.6f} ± {results['accuracy_std']:.6f}",
                    'Precision': f"{results['precision_mean']:.6f} ± {results['precision_std']:.6f}",
                    'Recall': f"{results['recall_mean']:.6f} ± {results['recall_std']:.6f}",
                    'F1-Score': f"{results['f1_mean']:.6f} ± {results['f1_std']:.6f}",
                    'AUC': f"{results['auc_mean']:.6f} ± {results['auc_std']:.6f}",
                    'Training_Time': f"{results['training_time']:.4f}s"
                }
                comparison_data.append(comparison_row)
                
                print(f"✓ {model_name} 训练完成")
                print(f"  准确率: {results['accuracy_mean']:.6f}")
                print(f"  精确率: {results['precision_mean']:.6f}")
                print(f"  召回率: {results['recall_mean']:.6f}")
                print(f"  F1分数: {results['f1_mean']:.6f}")
                print(f"  AUC: {results['auc_mean']:.6f}")
                print(f"  训练时间: {results['training_time']:.4f}秒")
                
            except Exception as e:
                print(f"✗ {model_name} 训练失败: {e}")
                continue
        
        # 创建对比结果DataFrame
        self.comparison_results = pd.DataFrame(comparison_data)
        
        print("\n" + "=" * 60)
        print("模型性能对比总结:")
        print("=" * 60)
        print(self.comparison_results.to_string(index=False))
        
        return self.comparison_results
    
    def ablation_study(self, X, y, base_model='XGBoost'):
        """
        消融实验：测试不同组件的贡献
        
        参数:
            X (pd.DataFrame): 特征矩阵
            y (pd.Series): 目标变量
            base_model (str): 基础模型
            
        返回:
            pd.DataFrame: 消融实验结果
        """
        print(f"开始消融实验，基础模型: {base_model}")
        print("=" * 60)
        
        ablation_results = []
        
        # 1. 基础模型（无优化）
        print("\n1. 基础模型（无优化）")
        print("-" * 40)
        base_results = self.train_single_model(base_model, X, y)
        ablation_results.append({
            'Configuration': 'Base Model',
            'F1-Score': base_results['f1_mean'],
            'AUC': base_results['auc_mean'],
            'Training_Time': base_results['training_time']
        })
        
        # 2. 仅使用类别权重
        print("\n2. 添加类别权重")
        print("-" * 40)
        # 计算类别权重
        class_counts = y.value_counts()
        scale_pos_weight = class_counts[0] / class_counts[1]
        
        weighted_results = self.train_single_model(base_model, X, y, 
                                                 {'scale_pos_weight': scale_pos_weight})
        ablation_results.append({
            'Configuration': 'Base + Class Weight',
            'F1-Score': weighted_results['f1_mean'],
            'AUC': weighted_results['auc_mean'],
            'Training_Time': weighted_results['training_time']
        })
        
        # 3. 使用采样后的数据
        print("\n3. 使用采样数据")
        print("-" * 40)
        from enhanced_data_processor import EnhancedDataProcessor
        processor = EnhancedDataProcessor()
        X_sampled, y_sampled = processor.hybrid_sampling(X, y, target_ratio=5)
        
        sampled_results = self.train_single_model(base_model, X_sampled, y_sampled)
        ablation_results.append({
            'Configuration': 'Base + Sampling',
            'F1-Score': sampled_results['f1_mean'],
            'AUC': sampled_results['auc_mean'],
            'Training_Time': sampled_results['training_time']
        })
        
        # 4. 使用增强特征
        print("\n4. 使用增强特征")
        print("-" * 40)
        # 这里假设X已经包含增强特征
        enhanced_results = self.train_single_model(base_model, X, y)
        ablation_results.append({
            'Configuration': 'Base + Enhanced Features',
            'F1-Score': enhanced_results['f1_mean'],
            'AUC': enhanced_results['auc_mean'],
            'Training_Time': enhanced_results['training_time']
        })
        
        # 5. 完整优化版本
        print("\n5. 完整优化版本")
        print("-" * 40)
        full_results = self.train_single_model(base_model, X_sampled, y_sampled,
                                             {'scale_pos_weight': scale_pos_weight})
        ablation_results.append({
            'Configuration': 'Full Optimization',
            'F1-Score': full_results['f1_mean'],
            'AUC': full_results['auc_mean'],
            'Training_Time': full_results['training_time']
        })
        
        # 创建消融实验结果DataFrame
        ablation_df = pd.DataFrame(ablation_results)
        
        print("\n" + "=" * 60)
        print("消融实验结果:")
        print("=" * 60)
        print(ablation_df.to_string(index=False))
        
        # 计算改进幅度
        base_f1 = ablation_results[0]['F1-Score']
        base_auc = ablation_results[0]['AUC']
        
        print("\n改进幅度分析:")
        print("-" * 40)
        for result in ablation_results[1:]:
            f1_improvement = ((result['F1-Score'] - base_f1) / base_f1) * 100
            auc_improvement = ((result['AUC'] - base_auc) / base_auc) * 100
            print(f"{result['Configuration']}:")
            print(f"  F1改进: {f1_improvement:+.2f}%")
            print(f"  AUC改进: {auc_improvement:+.2f}%")
        
        return ablation_df
    
    def get_best_model(self, metric='f1'):
        """
        获取最佳模型
        
        参数:
            metric (str): 评估指标 ('f1', 'auc', 'accuracy', 'precision', 'recall')
            
        返回:
            tuple: (最佳模型名称, 模型对象, 性能分数)
        """
        if not self.results:
            return None, None, None
        
        best_model_name = None
        best_score = -np.inf
        best_model = None
        
        metric_map = {
            'f1': 'f1_mean',
            'auc': 'auc_mean',
            'accuracy': 'accuracy_mean',
            'precision': 'precision_mean',
            'recall': 'recall_mean'
        }
        
        if metric not in metric_map:
            raise ValueError(f"不支持的指标: {metric}")
        
        metric_key = metric_map[metric]
        
        for model_name, results in self.results.items():
            if results[metric_key] > best_score:
                best_score = results[metric_key]
                best_model_name = model_name
                best_model = results['model']
        
        return best_model_name, best_model, best_score
    
    def generate_comparison_report(self):
        """
        生成详细的对比报告
        
        返回:
            str: 格式化的报告字符串
        """
        if not self.comparison_results.empty:
            report = []
            report.append("=" * 80)
            report.append("模型性能对比详细报告")
            report.append("=" * 80)
            
            report.append("\n📊 性能排名:")
            
            # 按F1分数排序
            sorted_results = self.comparison_results.copy()
            
            # 提取F1分数用于排序
            f1_scores = []
            for idx, row in sorted_results.iterrows():
                f1_str = row['F1-Score']
                f1_score = float(f1_str.split(' ± ')[0])
                f1_scores.append(f1_score)
            
            sorted_results['F1_Score_Num'] = f1_scores
            sorted_results = sorted_results.sort_values('F1_Score_Num', ascending=False)
            
            for i, (idx, row) in enumerate(sorted_results.iterrows(), 1):
                report.append(f"{i}. {row['Model']} - F1: {row['F1-Score']}")
            
            report.append("\n🎯 最佳模型分析:")
            best_model_name, best_model, best_score = self.get_best_model('f1')
            if best_model_name:
                report.append(f"最佳模型: {best_model_name}")
                report.append(f"最佳F1分数: {best_score:.6f}")
                
                # 获取最佳模型的详细参数
                if best_model_name in self.results:
                    params = self.results[best_model_name]['params']
                    report.append(f"最佳参数配置:")
                    for param, value in params.items():
                        report.append(f"  {param}: {value}")
            
            report.append("\n" + "=" * 80)
            
            return "\n".join(report)
        
        return "暂无对比结果"

# 测试函数
def test_model_comparator():
    """测试模型对比器"""
    print("测试模型性能对比器...")
    
    # 创建模拟数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 15
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    
    # 转换为DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y)
    
    # 测试模型对比
    comparator = ModelComparator(cv_folds=3)
    
    # 对比部分模型（为了快速测试）
    comparison_results = comparator.compare_models(X_df, y_series, 
                                                 model_names=['XGBoost', 'RandomForest', 'LogisticRegression'])
    
    # 生成报告
    report = comparator.generate_comparison_report()
    print("\n" + report)
    
    print("\n✓ 模型性能对比器测试完成")

if __name__ == "__main__":
    test_model_comparator()
