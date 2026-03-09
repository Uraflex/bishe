#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据处理模块
负责数据的加载、清洗、预处理和探索性分析

作者: 转换自Jupyter Notebook
日期: 2026年3月9日
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 设置非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """数据处理器类"""
    
    def __init__(self):
        """初始化数据处理器"""
        self.scaler = StandardScaler()
        
    def load_data(self, file_path):
        """
        加载信用卡交易数据集
        
        参数:
            file_path (str): 数据文件路径
            
        返回:
            pd.DataFrame: 加载的数据集
        """
        try:
            df = pd.read_csv(file_path)
            print(f"成功加载数据集，共 {len(df)} 条记录")
            return df
        except FileNotFoundError:
            print(f"错误: 找不到文件 {file_path}")
            print("请确保 'creditcard.csv' 文件在当前目录下")
            return None
        except Exception as e:
            print(f"加载数据时发生错误: {e}")
            return None
    
    def explore_data(self, df):
        """
        探索性数据分析
        
        参数:
            df (pd.DataFrame): 数据集
        """
        if df is None:
            return
            
        print("\n数据集基本信息:")
        print(f"- 数据形状: {df.shape}")
        print(f"- 列名: {list(df.columns)}")
        
        # 检查缺失值
        print("\n缺失值统计:")
        missing_values = df.isnull().sum()
        if missing_values.sum() == 0:
            print("✓ 数据集中没有缺失值")
        else:
            print(missing_values[missing_values > 0])
        
        # 数据类型信息
        print("\n数据类型信息:")
        df.info()
        
        # 统计描述
        print("\n数值特征统计描述:")
        print(df.describe())
        
        # 类别分布分析
        self.analyze_class_distribution(df)
        
    def analyze_class_distribution(self, df):
        """
        分析目标变量的类别分布
        
        参数:
            df (pd.DataFrame): 包含目标变量的数据集
        """
        if 'Class' not in df.columns:
            print("警告: 数据集中没有找到 'Class' 列")
            return
            
        print("\n类别分布分析:")
        class_counts = df['Class'].value_counts()
        class_percentages = df['Class'].value_counts(normalize=True) * 100
        
        for class_label in class_counts.index:
            count = class_counts[class_label]
            percentage = class_percentages[class_label]
            class_name = "正常交易" if class_label == 0 else "欺诈交易"
            print(f"- {class_name} (Class={class_label}): {count} 条 ({percentage:.4f}%)")
        
        # 可视化类别分布
        plt.figure(figsize=(8, 6))
        sns.countplot(x='Class', data=df)
        plt.title('交易类别分布', fontsize=14)
        plt.xlabel('类别 (0:正常, 1:欺诈)', fontsize=12)
        plt.ylabel('交易数量', fontsize=12)
        
        # 在柱状图上添加数值标签
        for i, count in enumerate(class_counts):
            plt.text(i, count + class_counts.max() * 0.01, str(count), 
                    ha='center', va='bottom', fontsize=12)
        
        plt.tight_layout()
        plt.show()
        
        # 计算不平衡比例
        if len(class_counts) == 2:
            imbalance_ratio = class_counts[0] / class_counts[1]
            print(f"\n数据不平衡比例: {imbalance_ratio:.2f}:1 (正常:欺诈)")
            print("⚠️  数据集严重不平衡，需要特殊处理")
    
    def preprocess_data(self, df):
        """
        数据预处理
        
        参数:
            df (pd.DataFrame): 原始数据集
            
        返回:
            tuple: (特征矩阵X, 目标变量y)
        """
        if df is None:
            return None, None
            
        print("\n开始数据预处理...")
        
        # 1. 分离特征和目标变量
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        print(f"分离特征和目标变量:")
        print(f"- 特征数量: {X.shape[1]}")
        print(f"- 样本数量: {X.shape[0]}")
        
        # 2. 特征缩放
        print("\n执行特征缩放...")
        
        # 检查需要缩放的列
        columns_to_scale = ['Amount', 'Time']
        existing_columns = [col for col in columns_to_scale if col in X.columns]
        
        if existing_columns:
            print(f"对以下列进行标准化处理: {existing_columns}")
            X[existing_columns] = self.scaler.fit_transform(X[existing_columns])
            print("✓ 特征缩放完成")
        else:
            print("未找到需要缩放的列 (Amount, Time)")
        
        # 3. 检查特征的基本信息
        print("\n预处理后的特征信息:")
        print(f"- 特征矩阵形状: {X.shape}")
        print(f"- 特征名称: {list(X.columns)}")
        print(f"- 特征数据类型:\n{X.dtypes.value_counts()}")
        
        return X, y
    
    def get_feature_statistics(self, X):
        """
        获取特征的详细统计信息
        
        参数:
            X (pd.DataFrame): 特征矩阵
            
        返回:
            pd.DataFrame: 特征统计信息
        """
        stats = X.describe().T
        stats['missing_values'] = X.isnull().sum()
        stats['missing_percentage'] = (X.isnull().sum() / len(X)) * 100
        return stats
    
    def detect_outliers(self, X, method='iqr'):
        """
        检测异常值
        
        参数:
            X (pd.DataFrame): 特征矩阵
            method (str): 异常值检测方法 ('iqr' 或 'zscore')
            
        返回:
            dict: 每个特征的异常值统计
        """
        outlier_stats = {}
        
        for column in X.select_dtypes(include=[np.number]).columns:
            if method == 'iqr':
                Q1 = X[column].quantile(0.25)
                Q3 = X[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = X[(X[column] < lower_bound) | (X[column] > upper_bound)]
                
            elif method == 'zscore':
                z_scores = np.abs((X[column] - X[column].mean()) / X[column].std())
                outliers = X[z_scores > 3]
            
            outlier_stats[column] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(X)) * 100
            }
        
        return outlier_stats

# 使用示例和测试函数
def test_data_processor():
    """测试数据处理器功能"""
    print("测试数据处理器...")
    
    # 创建测试数据
    test_data = {
        'Time': [0, 1, 2, 3, 4],
        'V1': [-1.36, 1.19, -1.36, -0.97, -1.16],
        'V2': [-0.07, 0.27, -1.34, -0.19, 0.88],
        'Amount': [149.62, 2.69, 378.66, 123.50, 69.99],
        'Class': [0, 0, 0, 0, 1]
    }
    
    df = pd.DataFrame(test_data)
    
    # 测试数据处理器
    processor = DataProcessor()
    processor.explore_data(df)
    X, y = processor.preprocess_data(df)
    
    print("✓ 数据处理器测试完成")

if __name__ == "__main__":
    test_data_processor()
