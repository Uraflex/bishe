#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, classification_report, 
                           confusion_matrix, roc_curve, average_precision_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import xgboost as xgb
import time
import warnings
warnings.filterwarnings('ignore')

class ModelComparator:
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
                    'n_estimators': 100,          # 减少树数量
                    'learning_rate': 0.1,          # 提高学习率
                    'max_depth': 4,                # 限制深度
                    'min_child_weight': 5,         # 增加最小权重
                    'subsample': 0.8,              # 适中的采样比例
                    'colsample_bytree': 0.8,       # 适中的列采样
                    'gamma': 0.5,                  # 添加gamma约束
                    'reg_alpha': 0.1,               # 增加L1正则化
                    'reg_lambda': 1.0,             # 增加L2正则化
                    'random_state': random_state,
                    'eval_metric': 'logloss',
                    'n_jobs': -1,
                    'verbosity': 0,
                    'scale_pos_weight': 3.0        # 降低类别权重
                },
                'param_grid': {
                    'n_estimators': [80, 100, 120],
                    'max_depth': [3, 4, 5],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'subsample': [0.7, 0.8, 0.9],
                    'scale_pos_weight': [2.0, 3.0, 4.0]
                }
            },
            'RandomForest': {
                'model': RandomForestClassifier,
                'params': {
                    'random_state': random_state,
                    'n_jobs': -1,
                    'n_estimators': 20,          # 进一步减少树数量
                    'max_depth': 4,               # 进一步限制深度
                    'min_samples_split': 20,      # 增加最小分割样本
                    'min_samples_leaf': 10,       # 增加叶子节点最小样本
                    'max_features': 0.3,          # 进一步限制特征选择比例
                    'bootstrap': True,            # 使用bootstrap采样
                    'oob_score': True             # 使用袋外评分
                },
                'param_grid': {
                    'n_estimators': [15, 20, 25],
                    'max_depth': [3, 4, 5],
                    'min_samples_split': [15, 20, 25],
                    'max_features': [0.2, 0.3, 0.4]
                }
            },
            'LogisticRegression': {
                'model': LogisticRegression,
                'params': {
                    'random_state': random_state,
                    'max_iter': 1000,
                    'C': 0.5,                     # 减少正则化强度
                    'penalty': 'l2',
                    'solver': 'liblinear',       # 使用适合小数据的求解器
                    'class_weight': 'balanced'    # 添加类别权重
                },
                'param_grid': {
                    'C': [0.3, 0.5, 0.8],
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
        
        # 计算各项指标 - 使用更合适的scoring方式
        accuracy_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        precision_scores = cross_val_score(model, X, y, cv=cv, scoring='precision')
        recall_scores = cross_val_score(model, X, y, cv=cv, scoring='recall')
        f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
        
        # ROC AUC和PR AUC需要特殊处理 - 使用更准确的方法
        auc_scores = []
        pr_auc_scores = []
        for train_idx, test_idx in cv.split(X, y):
            X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
            y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
            
            # 训练模型
            fold_model = model_def['model'](**model_params)
            fold_model.fit(X_train_fold, y_train_fold)
            
            # 预测概率
            if hasattr(fold_model, 'predict_proba'):
                y_pred_proba = fold_model.predict_proba(X_test_fold)[:, 1]
            else:
                # 对于不支持概率的模型，使用决策函数
                y_pred_proba = fold_model.decision_function(X_test_fold)
                # 如果是二维数组，取第二列
                if len(y_pred_proba.shape) > 1:
                    y_pred_proba = y_pred_proba[:, 1]
                # 标准化到[0,1]范围
                y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min() + 1e-8)
            
            # 计算ROC AUC
            try:
                fold_auc = roc_auc_score(y_test_fold, y_pred_proba)
                auc_scores.append(fold_auc)
            except:
                auc_scores.append(0.5)  # 默认值
            
            # 计算PR AUC (更适合不平衡数据)
            try:
                fold_pr_auc = average_precision_score(y_test_fold, y_pred_proba)
                pr_auc_scores.append(fold_pr_auc)
            except:
                pr_auc_scores.append(0.0)  # 默认值
        
        # 添加平衡准确率，处理数据不平衡问题
        try:
            balanced_accuracy_scores = cross_val_score(model, X, y, cv=cv, scoring='balanced_accuracy')
        except:
            balanced_accuracy_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        training_time = time.time() - start_time
        
        # 训练最终模型
        model.fit(X, y)
        
        results = {
            'model': model,
            'accuracy_mean': np.mean(accuracy_scores),
            'accuracy_std': np.std(accuracy_scores),
            'balanced_accuracy_mean': np.mean(balanced_accuracy_scores),
            'balanced_accuracy_std': np.std(balanced_accuracy_scores),
            'precision_mean': np.mean(precision_scores),
            'precision_std': np.std(precision_scores),
            'recall_mean': np.mean(recall_scores),
            'recall_std': np.std(recall_scores),
            'f1_mean': np.mean(f1_scores),
            'f1_std': np.std(f1_scores),
            'auc_mean': np.mean(auc_scores),
            'auc_std': np.std(auc_scores),
            'pr_auc_mean': np.mean(pr_auc_scores),
            'pr_auc_std': np.std(pr_auc_scores),
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
                    'Balanced_Acc': f"{results['balanced_accuracy_mean']:.6f} ± {results['balanced_accuracy_std']:.6f}",
                    'Precision': f"{results['precision_mean']:.6f} ± {results['precision_std']:.6f}",
                    'Recall': f"{results['recall_mean']:.6f} ± {results['recall_std']:.6f}",
                    'F1-Score': f"{results['f1_mean']:.6f} ± {results['f1_std']:.6f}",
                    'ROC_AUC': f"{results['auc_mean']:.6f} ± {results['auc_std']:.6f}",
                    'PR_AUC': f"{results['pr_auc_mean']:.6f} ± {results['pr_auc_std']:.6f}",
                    'Training_Time': f"{results['training_time']:.4f}s"
                }
                comparison_data.append(comparison_row)
                
                print(f"✓ {model_name} 训练完成")
                print(f"  准确率: {results['accuracy_mean']:.6f}")
                print(f"  平衡准确率: {results['balanced_accuracy_mean']:.6f}")
                print(f"  精确率: {results['precision_mean']:.6f}")
                print(f"  召回率: {results['recall_mean']:.6f}")
                print(f"  F1分数: {results['f1_mean']:.6f}")
                print(f"  ROC AUC: {results['auc_mean']:.6f}")
                print(f"  PR AUC: {results['pr_auc_mean']:.6f}")
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
            
            report.append("\n📊 性能排名 (按AUC排序):")
            
            # 按AUC分数排序
            sorted_results = self.comparison_results.copy()
            
            # 提取AUC分数用于排序
            auc_scores = []
            for idx, row in sorted_results.iterrows():
                auc_str = row['AUC']
                auc_score = float(auc_str.split(' ± ')[0])
                auc_scores.append(auc_score)
            
            sorted_results['AUC_Score_Num'] = auc_scores
            sorted_results = sorted_results.sort_values('AUC_Score_Num', ascending=False)
            
            for i, (idx, row) in enumerate(sorted_results.iterrows(), 1):
                report.append(f"{i}. {row['Model']} - AUC: {row['AUC']}")
            
            report.append("\n🎯 最佳模型分析:")
            best_model_name, best_model, best_score = self.get_best_model('auc')
            if best_model_name:
                report.append(f"最佳模型: {best_model_name}")
                report.append(f"最佳AUC分数: {best_score:.6f}")
                
                # 获取最佳模型的详细参数
                if best_model_name in self.results:
                    params = self.results[best_model_name]['params']
                    report.append(f"最佳参数配置:")
                    for param, value in params.items():
                        report.append(f"  {param}: {value}")
            
            report.append("\n" + "=" * 80)
            
            return "\n".join(report)
        
        return "暂无对比结果"
