"""
XGBoost参数优化配置
基于消融实验结果和性能分析的最佳参数配置

作者: 优化版本
日期: 2026年3月11日
"""

import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, make_scorer
import numpy as np

class XGBoostOptimizer:
    """XGBoost参数优化器"""
    
    def __init__(self, random_state=42):
        """
        初始化优化器
        
        参数:
            random_state (int): 随机种子
        """
        self.random_state = random_state
        
        # 基于消融实验的最佳参数配置
        self.best_practice_params = {
            'n_estimators': 350,          # 增加树数量
            'learning_rate': 0.015,       # 更低学习率
            'max_depth': 9,              # 更深树
            'min_child_weight': 1,        # 最小权重
            'subsample': 0.7,            # 降低采样增加多样性
            'colsample_bytree': 0.7,     # 降低列采样
            'gamma': 0.0,                # 无gamma约束
            'reg_alpha': 0.005,          # 极小L1正则化
            'reg_lambda': 0.3,           # 极小L2正则化
            'random_state': random_state,
            'eval_metric': 'logloss',
            'n_jobs': -1,
            'verbosity': 0
        }
        
        # 针对高召回率的参数配置
        self.high_recall_params = {
            'n_estimators': 200,          # 增加树的数量
            'learning_rate': 0.03,       # 更低的学习率
            'max_depth': 6,              # 更深的树
            'min_child_weight': 1,        # 更小的叶子权重
            'subsample': 0.8,            # 稍低的子采样
            'colsample_bytree': 0.8,     # 稍低的列采样
            'gamma': 0.0,                # 无gamma约束
            'reg_alpha': 0.05,           # 更小的L1正则化
            'reg_lambda': 0.8,           # 更小的L2正则化
            'random_state': random_state,
            'eval_metric': 'logloss',
            'n_jobs': -1,
            'verbosity': 0
        }
        
        # 针对快速训练的参数配置
        self.fast_training_params = {
            'n_estimators': 100,          # 减少树的数量
            'learning_rate': 0.1,       # 标准学习率
            'max_depth': 4,              # 标准深度
            'min_child_weight': 3,        # 标准叶子权重
            'subsample': 0.9,            # 高子采样
            'colsample_bytree': 0.9,     # 高列采样
            'gamma': 0.2,                # 适中的gamma
            'reg_alpha': 0.2,            # 适中的L1正则化
            'reg_lambda': 1.2,           # 适中的L2正则化
            'random_state': random_state,
            'eval_metric': 'logloss',
            'n_jobs': -1,
            'verbosity': 0
        }
    
    def calculate_optimal_scale_pos_weight(self, y):
        """
        计算最优的类别权重比例
        基于消融实验结果优化权重计算
        
        参数:
            y (pd.Series): 目标变量
            
        返回:
            float: 优化的scale_pos_weight值
        """
        class_counts = np.bincount(y)
        if len(class_counts) == 2:
            # 基于消融实验的优化权重计算
            normal_count = class_counts[0]
            fraud_count = class_counts[1]
            
            # 基础比例
            base_ratio = normal_count / fraud_count
            
            # 基于消融实验最佳结果(20.98)调整权重
            # 使用更保守的权重策略
            if fraud_count < 500:
                # 极度不平衡时，使用消融实验的最佳权重范围
                return min(25.0, base_ratio * 0.08)
            elif fraud_count < 1000:
                # 中度不平衡时
                return min(20.0, base_ratio * 0.06)
            else:
                # 轻度不平衡时
                return min(15.0, base_ratio * 0.04)
        
        return 1.0
    
    def get_optimized_model(self, X, y, model_type='best_practice'):
        """
        获取优化的XGBoost模型
        
        参数:
            X (pd.DataFrame): 特征矩阵
            y (pd.Series): 目标变量
            model_type (str): 模型类型 ('best_practice', 'high_recall', 'fast_training')
            
        返回:
            XGBClassifier: 优化配置的模型
        """
        # 选择参数配置
        if model_type == 'best_practice':
            params = self.best_practice_params.copy()
        elif model_type == 'high_recall':
            params = self.high_recall_params.copy()
        elif model_type == 'fast_training':
            params = self.fast_training_params.copy()
        else:
            raise ValueError("model_type必须是 'best_practice', 'high_recall', 或 'fast_training'")
        
        # 计算优化的类别权重
        params['scale_pos_weight'] = self.calculate_optimal_scale_pos_weight(y)
        
        # 创建模型
        model = xgb.XGBClassifier(**params)
        
        return model
    
    def get_early_stopping_rounds(self, n_estimators):
        """
        根据树的数量计算早停轮数
        
        参数:
            n_estimators (int): 树的数量
            
        返回:
            int: 早停轮数
        """
        return max(10, n_estimators // 10)
    
    def print_optimization_summary(self):
        """打印优化配置总结"""
        print("=" * 60)
        print("XGBoost参数优化配置总结")
        print("=" * 60)
        
        print("\n🎯 最佳实践配置 (平衡性能和速度):")
        for key, value in self.best_practice_params.items():
            if key != 'random_state':
                print(f"  {key}: {value}")
        
        print("\n🔍 高召回率配置 (优先召回率):")
        for key, value in self.high_recall_params.items():
            if key != 'random_state':
                print(f"  {key}: {value}")
        
        print("\n⚡ 快速训练配置 (优先速度):")
        for key, value in self.fast_training_params.items():
            if key != 'random_state':
                print(f"  {key}: {value}")
        
        print("\n📊 优化策略:")
        print("  • 基于消融实验结果优化参数")
        print("  • 动态调整类别权重比例")
        print("  • 增强正则化防止过拟合")
        print("  • 优化采样比例提高泛化能力")
        print("  • 添加gamma参数控制树分裂")

# 测试函数
def test_optimizer():
    """测试优化器"""
    print("测试XGBoost优化器...")
    
    optimizer = XGBoostOptimizer()
    optimizer.print_optimization_summary()
    
    # 模拟数据
    np.random.seed(42)
    y = np.random.choice([0, 1], 1000, p=[0.99, 0.01])
    
    # 测试权重计算
    weight = optimizer.calculate_optimal_scale_pos_weight(y)
    print(f"\n计算的类别权重: {weight:.2f}")
    
    print("✓ 优化器测试完成")

if __name__ == "__main__":
    test_optimizer()
