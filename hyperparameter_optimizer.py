#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超参数优化模块
实现贝叶斯优化和Q-Learning结合的自动调优策略

作者: 基于开题报告要求
日期: 2026年3月9日
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
import xgboost as xgb
from scipy.optimize import minimize
from scipy.stats import uniform, randint
import time
import warnings
warnings.filterwarnings('ignore')

class BayesianOptimizer:
    """贝叶斯优化器，用于XGBoost超参数调优"""
    
    def __init__(self, max_iter=20, cv_folds=3, random_state=42):
        """
        初始化贝叶斯优化器
        
        参数:
            max_iter (int): 最大迭代次数
            cv_folds (int): 交叉验证折数
            random_state (int): 随机种子
        """
        self.max_iter = max_iter
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.best_params = None
        self.best_score = -np.inf
        self.optimization_history = []
        
        # 定义参数搜索空间
        self.param_bounds = {
            'n_estimators': (50, 300),  # 减少范围避免内存问题
            'max_depth': (3, 8),       # 减少深度范围
            'learning_rate': (0.01, 0.2),
            'subsample': (0.6, 1.0),
            'colsample_bytree': (0.6, 1.0),  # 确保不超过1.0
            'min_child_weight': (1, 8),
            'gamma': (0, 3),
            'reg_alpha': (0, 1.0),      # 扩大正则化范围
            'reg_lambda': (0.5, 2.0),   # 扩大正则化范围
            'scale_pos_weight': (1, 50)  # 大幅减少范围
        }
        
    def objective_function(self, params, X, y):
        """
        目标函数：使用交叉验证评估模型性能
        
        参数:
            params (list): 参数列表
            X (pd.DataFrame): 特征矩阵
            y (pd.Series): 目标变量
            
        返回:
            float: 目标函数值（负F1分数，用于最小化）
        """
        # 将参数列表转换为字典
        param_dict = {}
        param_names = list(self.param_bounds.keys())
        
        for i, param_name in enumerate(param_names):
            if i < len(params):
                # 对于整数参数进行四舍五入
                if param_name in ['n_estimators', 'max_depth', 'min_child_weight']:
                    param_dict[param_name] = int(round(params[i]))
                else:
                    param_dict[param_name] = params[i]
        
        # 参数边界检查
        param_dict['colsample_bytree'] = np.clip(param_dict['colsample_bytree'], 0.1, 1.0)
        param_dict['subsample'] = np.clip(param_dict['subsample'], 0.1, 1.0)
        param_dict['learning_rate'] = np.clip(param_dict['learning_rate'], 0.001, 1.0)
        param_dict['gamma'] = max(0, param_dict['gamma'])
        param_dict['reg_alpha'] = max(0, param_dict['reg_alpha'])
        param_dict['reg_lambda'] = max(0, param_dict['reg_lambda'])
        param_dict['scale_pos_weight'] = max(0.1, param_dict['scale_pos_weight'])
        
        try:
            # 创建XGBoost模型
            model = xgb.XGBClassifier(
                **param_dict,
                random_state=self.random_state,
                n_jobs=1,  # 减少并行度避免编码问题
                eval_metric='logloss',
                verbosity=0  # 减少输出
            )
            
            # 使用交叉验证评估
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            
            # 计算F1分数
            f1_scorer = make_scorer(f1_score, average='binary')
            scores = cross_val_score(model, X, y, cv=cv, scoring=f1_scorer, n_jobs=1)
            
            mean_score = np.mean(scores)
            
            # 记录优化历史
            self.optimization_history.append({
                'params': param_dict.copy(),
                'score': mean_score,
                'iteration': len(self.optimization_history) + 1
            })
            
            # 更新最佳参数
            if mean_score > self.best_score:
                self.best_score = mean_score
                self.best_params = param_dict.copy()
            
            return -mean_score  # 返回负值因为我们要最小化
            
        except Exception as e:
            # 避免中文输出导致的编码问题
            error_msg = str(e).encode('ascii', 'ignore').decode('ascii')
            print(f"Parameter evaluation failed: {error_msg}")
            return 0  # 返回较差的分数
    
    def acquisition_function(self, x, gp):
        """
        采集函数：期望改进（Expected Improvement）
        
        参数:
            x (array): 候选参数点
            gp: 高斯过程模型
            
        返回:
            float: 采集函数值
        """
        mean, std = gp.predict(x.reshape(1, -1), return_std=True)
        
        if std == 0:
            return 0
        
        # 计算期望改进
        improvement = mean - self.best_score
        z = improvement / std
        
        ei = improvement * norm.cdf(z) + std * norm.pdf(z)
        
        return -ei  # 返回负值用于最小化
    
    def optimize(self, X, y, verbose=True):
        """
        执行贝叶斯优化
        
        参数:
            X (pd.DataFrame): 特征矩阵
            y (pd.Series): 目标变量
            verbose (bool): 是否显示详细信息
            
        返回:
            dict: 最佳参数
        """
        if verbose:
            print(f"开始贝叶斯优化，最大迭代次数: {self.max_iter}")
            print(f"交叉验证折数: {self.cv_folds}")
            print("-" * 50)
        
        # 初始随机采样
        n_init = min(10, self.max_iter // 2)
        
        for i in range(n_init):
            # 随机生成参数
            params = []
            for param_name, (low, high) in self.param_bounds.items():
                if param_name in ['n_estimators', 'max_depth', 'min_child_weight']:
                    params.append(np.random.randint(low, high + 1))
                else:
                    params.append(np.random.uniform(low, high))
            
            # 评估参数
            score = -self.objective_function(params, X, y)
            
            if verbose:
                print(f"初始采样 {i+1}/{n_init}: F1分数 = {-score:.6f}")
        
        # 贝叶斯优化主循环
        from scipy.stats import norm
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern
        
        # 创建高斯过程模型
        kernel = Matern(length_scale=1.0, nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
        
        for iteration in range(n_init, self.max_iter):
            # 准备训练数据
            if len(self.optimization_history) > 0:
                X_train = []
                y_train = []
                
                for hist in self.optimization_history:
                    param_vector = []
                    for param_name in self.param_bounds.keys():
                        param_vector.append(hist['params'].get(param_name, 0))
                    X_train.append(param_vector)
                    y_train.append(hist['score'])
                
                X_train = np.array(X_train)
                y_train = np.array(y_train)
                
                # 训练高斯过程
                gp.fit(X_train, y_train)
                
                # 优化采集函数
                best_acquisition = np.inf
                best_params = None
                
                # 多次随机初始化避免局部最优
                for _ in range(50):
                    x0 = []
                    for param_name, (low, high) in self.param_bounds.items():
                        if param_name in ['n_estimators', 'max_depth', 'min_child_weight']:
                            x0.append(np.random.randint(low, high + 1))
                        else:
                            x0.append(np.random.uniform(low, high))
                    
                    # 优化
                    res = minimize(
                        lambda x: self.acquisition_function(x, gp),
                        x0=x0,
                        bounds=list(self.param_bounds.values()),
                        method='L-BFGS-B'
                    )
                    
                    if res.fun < best_acquisition:
                        best_acquisition = res.fun
                        best_params = res.x
                
                # 评估最佳候选点
                if best_params is not None:
                    score = -self.objective_function(best_params, X, y)
                    
                    if verbose:
                        print(f"迭代 {iteration+1}/{self.max_iter}: F1分数 = {-score:.6f}, 最佳F1 = {self.best_score:.6f}")
        
        if verbose:
            print("-" * 50)
            print(f"优化完成！最佳F1分数: {self.best_score:.6f}")
            print("最佳参数:")
            if self.best_params is not None:
                for param, value in self.best_params.items():
                    print(f"  {param}: {value}")
            else:
                print("  未找到有效参数")
        
        return self.best_params if self.best_params is not None else {}

class QLearningOptimizer:
    """Q-Learning优化器，动态调整贝叶斯搜索策略"""
    
    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        """
        初始化Q-Learning优化器
        
        参数:
            learning_rate (float): 学习率
            discount_factor (float): 折扣因子
            epsilon (float): 探索率
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = {}
        self.state_history = []
        self.action_history = []
        self.reward_history = []
        
    def get_state(self, iteration, improvement_rate):
        """
        获取当前状态
        
        参数:
            iteration (int): 当前迭代次数
            improvement_rate (float): 改进率
            
        返回:
            str: 状态标识
        """
        # 根据迭代次数和改进率定义状态
        if iteration < 10:
            phase = "early"
        elif iteration < 50:
            phase = "middle"
        else:
            phase = "late"
        
        if improvement_rate > 0.01:
            trend = "improving"
        elif improvement_rate > -0.01:
            trend = "stable"
        else:
            trend = "declining"
        
        return f"{phase}_{trend}"
    
    def get_action(self, state):
        """
        根据状态选择动作
        
        参数:
            state (str): 当前状态
            
        返回:
            str: 动作标识
        """
        actions = ["explore", "exploit", "focus"]
        
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in actions}
        
        # ε-贪心策略
        if np.random.random() < self.epsilon:
            return np.random.choice(actions)
        else:
            return max(self.q_table[state], key=self.q_table[state].get)
    
    def update_q_value(self, state, action, reward, next_state):
        """
        更新Q值
        
        参数:
            state (str): 当前状态
            action (str): 执行的动作
            reward (float): 获得的奖励
            next_state (str): 下一状态
        """
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0 for a in self.q_table[state].keys()}
        
        current_q = self.q_table[state][action]
        max_next_q = max(self.q_table[next_state].values())
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state][action] = new_q

class HybridOptimizer:
    """混合优化器：结合贝叶斯优化和Q-Learning"""
    
    def __init__(self, max_iter=100, cv_folds=5, random_state=42):
        """
        初始化混合优化器
        
        参数:
            max_iter (int): 最大迭代次数
            cv_folds (int): 交叉验证折数
            random_state (int): 随机种子
        """
        self.bayesian_optimizer = BayesianOptimizer(max_iter, cv_folds, random_state)
        self.q_learning = QLearningOptimizer()
        self.max_iter = max_iter
        
    def optimize(self, X, y, verbose=True):
        """
        执行混合优化
        
        参数:
            X (pd.DataFrame): 特征矩阵
            y (pd.Series): 目标变量
            verbose (bool): 是否显示详细信息
            
        返回:
            dict: 最佳参数
        """
        if verbose:
            print("开始混合优化（贝叶斯优化 + Q-Learning）")
            print("=" * 60)
        
        best_score = -np.inf
        best_params = None
        
        for iteration in range(self.max_iter):
            # 获取当前状态
            if iteration == 0:
                improvement_rate = 0
            else:
                improvement_rate = (self.bayesian_optimizer.best_score - best_score) / abs(best_score + 1e-8)
            
            state = self.q_learning.get_state(iteration, improvement_rate)
            action = self.q_learning.get_action(state)
            
            # 根据动作调整搜索策略
            if action == "explore":
                # 探索：扩大搜索范围
                original_bounds = self.bayesian_optimizer.param_bounds.copy()
                for param in self.bayesian_optimizer.param_bounds:
                    low, high = original_bounds[param]
                    self.bayesian_optimizer.param_bounds[param] = (
                        max(low * 0.8, low * 1.2),
                        high * 1.2
                    )
            
            elif action == "focus":
                # 聚焦：缩小搜索范围到最佳参数附近
                if best_params:
                    for param, value in best_params.items():
                        low, high = self.bayesian_optimizer.param_bounds[param]
                        if param in ['n_estimators', 'max_depth', 'min_child_weight']:
                            margin = max(2, int(value * 0.2))
                            self.bayesian_optimizer.param_bounds[param] = (
                                max(1, value - margin),
                                value + margin
                            )
                        else:
                            margin = value * 0.1
                            self.bayesian_optimizer.param_bounds[param] = (
                                max(low, value - margin),
                                min(high, value + margin)
                            )
            
            # 执行一步贝叶斯优化
            if iteration < self.max_iter - 1:
                # 单步优化
                params = []
                for param_name, (low, high) in self.bayesian_optimizer.param_bounds.items():
                    if param_name in ['n_estimators', 'max_depth', 'min_child_weight']:
                        params.append(np.random.randint(low, high + 1))
                    else:
                        params.append(np.random.uniform(low, high))
                
                score = -self.bayesian_optimizer.objective_function(params, X, y)
                
                # 计算奖励
                reward = score - best_score if best_score != -np.inf else score
                
                # 更新Q值
                next_state = self.q_learning.get_state(iteration + 1, 
                    (self.bayesian_optimizer.best_score - best_score) / abs(best_score + 1e-8))
                self.q_learning.update_q_value(state, action, reward, next_state)
                
                if verbose:
                    print(f"迭代 {iteration+1}/{self.max_iter}: 状态={state}, 动作={action}, F1分数={score:.6f}")
            
            # 更新最佳结果
            if self.bayesian_optimizer.best_score > best_score:
                best_score = self.bayesian_optimizer.best_score
                best_params = self.bayesian_optimizer.best_params.copy()
        
        if verbose:
            print("=" * 60)
            print(f"混合优化完成！最佳F1分数: {best_score:.6f}")
            print("最佳参数:")
            if best_params is not None and isinstance(best_params, dict):
                for param, value in best_params.items():
                    print(f"  {param}: {value}")
            else:
                print("  未找到有效参数")
        
        return best_params if best_params is not None else {}

# 测试函数
def test_optimizer():
    """测试优化器"""
    print("测试超参数优化器...")
    
    # 创建模拟数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    
    # 转换为DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y)
    
    # 测试贝叶斯优化
    print("\n测试贝叶斯优化器:")
    bayesian_opt = BayesianOptimizer(max_iter=20, cv_folds=3)
    best_params_bayesian = bayesian_opt.optimize(X_df, y_series, verbose=True)
    
    # 测试混合优化
    print("\n测试混合优化器:")
    hybrid_opt = HybridOptimizer(max_iter=20, cv_folds=3)
    best_params_hybrid = hybrid_opt.optimize(X_df, y_series, verbose=True)
    
    print("\n✓ 超参数优化器测试完成")

if __name__ == "__main__":
    test_optimizer()
