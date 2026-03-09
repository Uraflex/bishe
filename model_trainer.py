#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型训练模块
负责XGBoost模型的训练、调优和保存

作者: 转换自Jupyter Notebook
日期: 2026年3月9日
"""

import pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import make_scorer, f1_score
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """模型训练器类"""
    
    def __init__(self):
        """初始化模型训练器"""
        self.model = None
        self.best_params = None
        
    def train_xgboost(self, X_train, y_train, use_grid_search=False):
        """
        训练XGBoost分类器
        
        参数:
            X_train (pd.DataFrame): 训练特征
            y_train (pd.Series): 训练标签
            use_grid_search (bool): 是否使用网格搜索调优参数
            
        返回:
            XGBClassifier: 训练好的模型
        """
        print("开始训练XGBoost模型...")
        
        if use_grid_search:
            print("使用网格搜索进行超参数调优...")
            model = self._train_with_grid_search(X_train, y_train)
        else:
            print("使用预设参数训练模型...")
            model = self._train_with_default_params(X_train, y_train)
        
        self.model = model
        print("✓ 模型训练完成")
        
        return model
    
    def _train_with_default_params(self, X_train, y_train):
        """
        使用默认参数训练XGBoost模型
        
        参数:
            X_train (pd.DataFrame): 训练特征
            y_train (pd.Series): 训练标签
            
        返回:
            XGBClassifier: 训练好的模型
        """
        # 针对不平衡数据集的XGBoost参数配置
        model = XGBClassifier(
            n_estimators=200,          # 树的数量
            learning_rate=0.1,         # 学习率
            max_depth=4,               # 树的最大深度
            subsample=0.8,            # 子采样比例
            colsample_bytree=0.8,     # 列采样比例
            random_state=42,           # 随机种子
            eval_metric='logloss',     # 评估指标
            # 处理不平衡数据的关键参数
            scale_pos_weight=self._calculate_scale_pos_weight(y_train),
            # 正则化参数防止过拟合
            reg_alpha=0.1,             # L1正则化
            reg_lambda=1.0,           # L2正则化
        )
        
        # 训练模型
        model.fit(X_train, y_train, verbose=False)
        
        # 打印模型参数
        print(f"模型参数配置:")
        print(f"- 树的数量: {model.n_estimators}")
        print(f"- 学习率: {model.learning_rate}")
        print(f"- 最大深度: {model.max_depth}")
        print(f"- 类别权重比例: {model.scale_pos_weight:.2f}")
        
        return model
    
    def _train_with_grid_search(self, X_train, y_train):
        """
        使用网格搜索调优XGBoost参数
        
        参数:
            X_train (pd.DataFrame): 训练特征
            y_train (pd.Series): 训练标签
            
        返回:
            XGBClassifier: 调优后的模型
        """
        # 定义参数网格
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 4, 5],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'scale_pos_weight': [self._calculate_scale_pos_weight(y_train)]
        }
        
        # 创建基础模型
        base_model = XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            reg_alpha=0.1,
            reg_lambda=1.0
        )
        
        # 定义评分器（针对不平衡数据，使用F1分数）
        f1_scorer = make_scorer(f1_score, pos_label=1)
        
        # 网格搜索
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring=f1_scorer,
            cv=3,                    # 3折交叉验证
            n_jobs=-1,               # 使用所有CPU核心
            verbose=1
        )
        
        # 执行网格搜索
        grid_search.fit(X_train, y_train)
        
        # 保存最佳参数
        self.best_params = grid_search.best_params_
        
        print(f"最佳参数: {self.best_params}")
        print(f"最佳F1分数: {grid_search.best_score_:.6f}")
        
        return grid_search.best_estimator_
    
    def _calculate_scale_pos_weight(self, y_train):
        """
        计算XGBoost的scale_pos_weight参数
        用于处理类别不平衡问题
        
        参数:
            y_train (pd.Series): 训练标签
            
        返回:
            float: 类别权重比例
        """
        # 计算负类（正常）和正类（欺诈）的样本数量
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        
        if pos_count == 0:
            return 1.0
        
        # XGBoost推荐的比例：负类样本数 / 正类样本数
        scale_pos_weight = neg_count / pos_count
        
        print(f"类别不平衡统计:")
        print(f"- 正常交易 (Class=0): {neg_count}")
        print(f"- 欺诈交易 (Class=1): {pos_count}")
        print(f"- 计算的权重比例: {scale_pos_weight:.2f}")
        
        return scale_pos_weight
    
    def cross_validate_model(self, X_train, y_train, cv=5):
        """
        对模型进行交叉验证
        
        参数:
            X_train (pd.DataFrame): 训练特征
            y_train (pd.Series): 训练标签
            cv (int): 交叉验证折数
            
        返回:
            dict: 交叉验证结果
        """
        if self.model is None:
            print("错误: 请先训练模型")
            return None
        
        print(f"执行{cv}折交叉验证...")
        
        # 定义多个评估指标
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'roc_auc': 'roc_auc'
        }
        
        cv_results = {}
        
        for metric_name, metric_scorer in scoring.items():
            scores = cross_val_score(
                self.model, X_train, y_train, 
                cv=cv, scoring=metric_scorer, n_jobs=-1
            )
            cv_results[metric_name] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores.tolist()
            }
            
            print(f"{metric_name.upper()}: {scores.mean():.6f} (±{scores.std():.6f})")
        
        return cv_results
    
    def save_model(self, model, file_path):
        """
        保存训练好的模型
        
        参数:
            model: 训练好的模型
            file_path (str): 模型保存路径
        """
        try:
            with open(file_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"模型已保存到: {file_path}")
        except Exception as e:
            print(f"保存模型时发生错误: {e}")
    
    def load_model(self, file_path):
        """
        加载已保存的模型
        
        参数:
            file_path (str): 模型文件路径
            
        返回:
            加载的模型
        """
        try:
            with open(file_path, 'rb') as f:
                model = pickle.load(f)
            print(f"模型已从 {file_path} 加载")
            self.model = model
            return model
        except FileNotFoundError:
            print(f"错误: 找不到模型文件 {file_path}")
            return None
        except Exception as e:
            print(f"加载模型时发生错误: {e}")
            return None
    
    def get_model_info(self):
        """
        获取模型信息
        
        返回:
            dict: 模型信息
        """
        if self.model is None:
            return {"error": "模型未训练"}
        
        info = {
            "model_type": type(self.model).__name__,
            "n_features": self.model.n_features_in_ if hasattr(self.model, 'n_features_in_') else None,
            "feature_names": self.model.get_booster().feature_names if hasattr(self.model, 'get_booster') else None,
            "n_estimators": getattr(self.model, 'n_estimators', None),
            "max_depth": getattr(self.model, 'max_depth', None),
            "learning_rate": getattr(self.model, 'learning_rate', None),
            "best_params": self.best_params
        }
        
        return info

# 测试函数
def test_model_trainer():
    """测试模型训练器功能"""
    print("测试模型训练器...")
    
    # 创建模拟数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    # 创建不平衡的标签
    y = np.random.choice([0, 1], size=n_samples, p=[0.98, 0.02])
    
    X_train = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    y_train = pd.Series(y)
    
    # 测试训练器
    trainer = ModelTrainer()
    model = trainer.train_xgboost(X_train, y_train)
    
    # 测试交叉验证
    cv_results = trainer.cross_validate_model(X_train, y_train, cv=3)
    
    print("✓ 模型训练器测试完成")

if __name__ == "__main__":
    test_model_trainer()
