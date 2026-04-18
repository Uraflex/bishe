#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class EnhancedDataProcessor:
    """数据处理器：混合采样、特征工程"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_stats = {}
        
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
            return None
        except Exception as e:
            print(f"加载数据时发生错误: {e}")
            return None
    
    def detect_outliers_iqr(self, df, features=None):
        """
        使用IQR法则检测异常值
        
        参数:
            df (pd.DataFrame): 数据集
            features (list): 需要检测的特征列表，None表示所有数值特征
            
        返回:
            pd.DataFrame: 清除异常值后的数据集
        """
        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'Class' in features:
                features.remove('Class')
        
        print(f"使用IQR法则检测异常值，检测特征: {features}")
        
        outlier_mask = pd.Series([False] * len(df))
        
        for feature in features:
            Q1 = df[feature].quantile(0.25)
            Q3 = df[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            feature_outliers = (df[feature] < lower_bound) | (df[feature] > upper_bound)
            outlier_mask |= feature_outliers
            
            print(f"  {feature}: 检测到 {feature_outliers.sum()} 个异常值")
        
        # 移除异常值，但保留所有欺诈样本（Class=1）
        fraud_mask = df['Class'] == 1
        final_mask = ~(outlier_mask & ~fraud_mask)
        
        cleaned_df = df[final_mask].copy()
        print(f"异常值处理完成，保留 {len(cleaned_df)} 条记录")
        
        return cleaned_df
    
    def create_advanced_features(self, df):
        """
        创建高级特征工程
        
        参数:
            df (pd.DataFrame): 原始数据集
            
        返回:
            pd.DataFrame: 包含新特征的数据集
        """
        print("开始高级特征工程...")
        
        df_enhanced = df.copy()
        
        # 1. 交易金额统计特征
        if 'Amount' in df.columns:
            # 金额分箱特征
            df_enhanced['Amount_bin'] = pd.cut(df['Amount'], 
                                             bins=[0, 10, 50, 100, 500, np.inf],
                                             labels=[0, 1, 2, 3, 4]).astype(float)
            
            # 金额对数变换
            df_enhanced['Amount_log'] = np.log1p(df['Amount'])
            
            # 金额标准化分数
            amount_mean = df['Amount'].mean()
            amount_std = df['Amount'].std()
            df_enhanced['Amount_zscore'] = (df['Amount'] - amount_mean) / amount_std
        
        # 2. 时间序列特征
        if 'Time' in df.columns:
            # 时间周期特征（假设Time是秒数）
            df_enhanced['Time_hour'] = (df['Time'] / 3600) % 24
            df_enhanced['Time_day'] = (df['Time'] / (3600 * 24)) % 7
            
            # 时间分箱
            df_enhanced['Time_hour_bin'] = pd.cut(df_enhanced['Time_hour'],
                                                  bins=[0, 6, 12, 18, 24],
                                                  labels=[0, 1, 2, 3]).astype(float)
        
        # 3. V特征统计特征
        v_features = [col for col in df.columns if col.startswith('V')]
        if v_features:
            # V特征的统计量
            df_enhanced['V_mean'] = df[v_features].mean(axis=1)
            df_enhanced['V_std'] = df[v_features].std(axis=1)
            df_enhanced['V_max'] = df[v_features].max(axis=1)
            df_enhanced['V_min'] = df[v_features].min(axis=1)
            df_enhanced['V_range'] = df_enhanced['V_max'] - df_enhanced['V_min']
            
            # 变异系数
            df_enhanced['V_cv'] = df_enhanced['V_std'] / (df_enhanced['V_mean'] + 1e-8)
            
            # 绝对值和
            df_enhanced['V_abs_sum'] = df[v_features].abs().sum(axis=1)
            
            # 正负特征数量
            df_enhanced['V_positive_count'] = (df[v_features] > 0).sum(axis=1)
            df_enhanced['V_negative_count'] = (df[v_features] < 0).sum(axis=1)
        
        # 4. 交互特征
        if 'Amount' in df.columns and v_features:
            # 金额与V特征的交互
            for i, v_feat in enumerate(v_features[:5]):  # 只取前5个避免特征过多
                df_enhanced[f'Amount_{v_feat}_interaction'] = df['Amount'] * df[v_feat]
        
        print(f"特征工程完成，新增 {len(df_enhanced.columns) - len(df.columns)} 个特征")
        print(f"总特征数: {len(df_enhanced.columns)}")
        
        return df_enhanced
    
    def hybrid_sampling(self, X, y, target_ratio=5, random_state=42):
        """
        混合采样策略：K-Means欠采样 + SMOTE过采样
        
        参数:
            X (pd.DataFrame): 特征矩阵
            y (pd.Series): 目标变量
            target_ratio (int): 目标正负样本比例（负:正）
            random_state (int): 随机种子
            
        返回:
            tuple: (采样后的特征矩阵, 采样后的目标变量)
        """
        print(f"开始混合采样，目标比例 {target_ratio}:1...")
        
        # 分离多数类和少数类
        majority_mask = (y == 0)
        minority_mask = (y == 1)
        
        X_majority = X[majority_mask]
        X_minority = X[minority_mask]
        y_majority = y[majority_mask]
        y_minority = y[minority_mask]

        # 处理NaN值：删除包含NaN的行
        X_majority = X_majority.dropna()
        y_majority = y_majority[X_majority.index]
        X_minority = X_minority.dropna()
        y_minority = y_minority[X_minority.index]
        
        print(f"原始样本分布 - 正常: {len(X_majority)}, 欺诈: {len(X_minority)}")
        
        # 1. K-Means聚类欠采样（多数类）
        # 减少聚类数量以提高性能
        n_clusters = min(100, len(X_minority) * target_ratio // 10)  # 大幅减少聚类数
        n_clusters = max(n_clusters, 10)  # 至少10个聚类
        
        print(f"对多数类进行K-Means聚类，聚类数: {n_clusters}")
        
        # 使用更快的K-Means配置
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=5, max_iter=100)
        cluster_labels = kmeans.fit_predict(X_majority)
        
        # 从每个聚类中随机选择样本
        undersampled_indices = []
        samples_per_cluster = len(X_minority) * target_ratio // n_clusters
        
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_indices) > 0:
                n_samples = min(samples_per_cluster, len(cluster_indices))
                selected = np.random.choice(cluster_indices, n_samples, replace=False)
                undersampled_indices.extend(selected)
        
        X_majority_undersampled = X_majority.iloc[undersampled_indices]
        y_majority_undersampled = y_majority.iloc[undersampled_indices]
        
        print(f"欠采样后多数类样本数: {len(X_majority_undersampled)}")
        
        # 2. SMOTE过采样（少数类）
        print("对少数类进行SMOTE过采样...")
        
        # 合并欠采样后的多数类和原始少数类
        X_combined = pd.concat([X_majority_undersampled, X_minority], ignore_index=True)
        y_combined = pd.concat([y_majority_undersampled, y_minority], ignore_index=True)
        
        # 计算需要的过采样数量
        target_minority_count = len(X_majority_undersampled) // target_ratio
        current_minority_count = len(X_minority)
        
        if current_minority_count < target_minority_count:
            # 使用SMOTE进行过采样
            sampling_strategy = {1: target_minority_count}
            smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
            X_resampled, y_resampled = smote.fit_resample(X_combined, y_combined)
        else:
            X_resampled, y_resampled = X_combined, y_combined
        
        # 3. 最终比例检查和调整
        final_majority_count = np.sum(y_resampled == 0)
        final_minority_count = np.sum(y_resampled == 1)
        final_ratio = final_majority_count / final_minority_count
        
        print(f"混合采样完成:")
        print(f"  正常样本: {final_majority_count}")
        print(f"  欺诈样本: {final_minority_count}")
        print(f"  实际比例: {final_ratio:.2f}:1")
        
        return X_resampled, y_resampled
    
    def preprocess_data_enhanced(self, df, use_hybrid_sampling=True, 
                                use_advanced_features=True, target_ratio=5):
        """
        增强数据预处理流程
        
        参数:
            df (pd.DataFrame): 原始数据集
            use_hybrid_sampling (bool): 是否使用混合采样
            use_advanced_features (bool): 是否使用高级特征工程
            target_ratio (int): 目标正负样本比例
            
        返回:
            tuple: (特征矩阵X, 目标变量y)
        """
        if df is None:
            return None, None
        
        print("开始增强数据预处理...")
        
        # 1. 异常值检测和处理
        print("\n1. 异常值检测和处理")
        print("-" * 30)
        df_cleaned = self.detect_outliers_iqr(df)
        
        # 2. 高级特征工程
        if use_advanced_features:
            print("\n2. 高级特征工程")
            print("-" * 30)
            df_enhanced = self.create_advanced_features(df_cleaned)
        else:
            df_enhanced = df_cleaned
        
        # 3. 分离特征和目标变量
        print("\n3. 分离特征和目标变量")
        print("-" * 30)
        X = df_enhanced.drop('Class', axis=1)
        y = df_enhanced['Class']
        
        print(f"特征数量: {X.shape[1]}")
        print(f"样本数量: {X.shape[0]}")
        
        # 4. 特征缩放
        print("\n4. 特征缩放")
        print("-" * 30)
        
        # 检查需要缩放的列
        columns_to_scale = ['Amount', 'Time']
        existing_columns = [col for col in columns_to_scale if col in X.columns]
        
        if existing_columns:
            print(f"对以下列进行标准化处理: {existing_columns}")
            X[existing_columns] = self.scaler.fit_transform(X[existing_columns])
            print("✓ 特征缩放完成")
        
        # 5. 混合采样
        if use_hybrid_sampling:
            print("\n5. 混合采样")
            print("-" * 30)
            X_resampled, y_resampled = self.hybrid_sampling(X, y, target_ratio)
            return X_resampled, y_resampled
        else:
            return X, y
    
    def preprocess_data_split(self, df, test_size=0.2, use_hybrid_sampling=True, 
                               use_advanced_features=True, target_ratio=5, random_state=42):
        """
        正确的数据预处理流程（避免数据泄露）
        
        流程：
        1. 先做特征工程（全局特征，不涉及统计量泄露）
        2. 分割训练集和测试集
        3. 分别对训练集和测试集进行标准化（使用训练集的统计量）
        4. 只对训练集进行混合采样
        
        参数:
            df (pd.DataFrame): 原始数据集
            test_size (float): 测试集比例
            use_hybrid_sampling (bool): 是否使用混合采样
            use_advanced_features (bool): 是否使用高级特征工程
            target_ratio (int): 目标正负样本比例
            random_state (int): 随机种子
            
        返回:
            tuple: (X_train, X_test, y_train, y_test)
        """
        if df is None:
            return None, None, None, None
        
        print("开始正确的数据预处理流程（无数据泄露）...")
        
        # 1. 异常值检测和处理（在全局做，不影响统计分布）
        print("\n1. 异常值检测和处理")
        print("-" * 30)
        df_cleaned = self.detect_outliers_iqr(df)
        
        # 2. 高级特征工程（使用保守模式减少过拟合）
        if use_advanced_features:
            print("\n2. 高级特征工程（保守模式，减少过拟合风险）")
            print("-" * 30)
            df_enhanced = self.create_advanced_features_v2(df_cleaned, conservative=True)
        else:
            df_enhanced = df_cleaned
        
        # 3. 分离特征和目标变量
        print("\n3. 分离特征和目标变量")
        print("-" * 30)
        X = df_enhanced.drop('Class', axis=1)
        y = df_enhanced['Class']
        
        print(f"特征数量: {X.shape[1]}")
        print(f"样本数量: {X.shape[0]}")
        
        # 4. 先分割训练集和测试集（关键步骤，防止数据泄露）
        print("\n4. 分割训练集和测试集")
        print("-" * 30)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
        
        # 5. 特征缩放（只在训练集上fit，然后transform训练集和测试集）
        print("\n5. 特征缩放（无数据泄露）")
        print("-" * 30)
        
        columns_to_scale = ['Amount', 'Time']
        existing_columns = [col for col in columns_to_scale if col in X_train.columns]
        
        if existing_columns:
            print(f"使用训练集统计量标准化: {existing_columns}")
            # 在训练集上fit
            self.scaler.fit(X_train[existing_columns])
            # transform训练集和测试集
            X_train[existing_columns] = self.scaler.transform(X_train[existing_columns])
            X_test[existing_columns] = self.scaler.transform(X_test[existing_columns])
            print("✓ 特征缩放完成")
        
        # 6. 只对训练集进行混合采样（关键：测试集保持原始分布）
        if use_hybrid_sampling:
            print("\n6. 对训练集进行混合采样（测试集保持不变）")
            print("-" * 30)
            X_train_resampled, y_train_resampled = self.hybrid_sampling(
                X_train, y_train, target_ratio, random_state
            )
            return X_train_resampled, X_test, y_train_resampled, y_test
        else:
            return X_train, X_test, y_train, y_test
    
    def create_advanced_features_v2(self, df, conservative=True):
        """
        创建高级特征工程（安全版本，避免使用全局统计量）
        
        避免使用基于整个数据集的统计量（如mean, std），
        只创建基于单条记录的特征，防止数据泄露
        
        参数:
            df (pd.DataFrame): 原始数据集
            conservative (bool): 是否使用保守模式（移除可能过强的V特征统计）
            
        返回:
            pd.DataFrame: 包含新特征的数据集
        """
        if conservative:
            print("开始安全特征工程（保守模式，减少过拟合风险）...")
        else:
            print("开始安全特征工程（完整模式）...")
        
        df_enhanced = df.copy()
        
        # 1. 交易金额特征（基于单条记录，无全局统计）
        if 'Amount' in df.columns:
            # 金额分箱特征（使用固定阈值，非数据驱动）
            df_enhanced['Amount_bin'] = pd.cut(df['Amount'], 
                                             bins=[0, 10, 50, 100, 500, np.inf],
                                             labels=[0, 1, 2, 3, 4]).astype(float)
            
            # 金额对数变换（单调变换，不改变分布形状）
            df_enhanced['Amount_log'] = np.log1p(df['Amount'])
        
        # 2. 时间序列特征
        if 'Time' in df.columns:
            # 时间周期特征
            df_enhanced['Time_hour'] = (df['Time'] / 3600) % 24
            df_enhanced['Time_day'] = (df['Time'] / (3600 * 24)) % 7
            
            # 时间分箱（固定阈值）
            df_enhanced['Time_hour_bin'] = pd.cut(df_enhanced['Time_hour'],
                                                  bins=[0, 6, 12, 18, 24],
                                                  labels=[0, 1, 2, 3]).astype(float)
        
        # 3. V特征统计特征（保守模式下减少统计特征数量）
        v_features = [col for col in df.columns if col.startswith('V')]
        if v_features:
            if conservative:
                # 保守模式：只保留最基本的统计量，减少过拟合
                df_enhanced['V_abs_sum'] = df[v_features].abs().sum(axis=1)
                print(f"  保守模式：仅添加 V_abs_sum（V特征绝对值和）")
            else:
                # 完整模式：添加所有V特征统计量
                df_enhanced['V_mean'] = df[v_features].mean(axis=1)
                df_enhanced['V_std'] = df[v_features].std(axis=1)
                df_enhanced['V_max'] = df[v_features].max(axis=1)
                df_enhanced['V_min'] = df[v_features].min(axis=1)
                df_enhanced['V_range'] = df_enhanced['V_max'] - df_enhanced['V_min']
                df_enhanced['V_cv'] = df_enhanced['V_std'] / (df_enhanced['V_mean'].abs() + 1e-8)
                df_enhanced['V_abs_sum'] = df[v_features].abs().sum(axis=1)
                df_enhanced['V_positive_count'] = (df[v_features] > 0).sum(axis=1)
                df_enhanced['V_negative_count'] = (df[v_features] < 0).sum(axis=1)
                print(f"  完整模式：添加所有V特征统计量")
        
        # 4. 交互特征（保守模式下减少交互特征）
        if 'Amount' in df.columns and v_features:
            if conservative:
                # 保守模式：只添加Amount与V1的交互
                df_enhanced['Amount_V1_interaction'] = df['Amount'] * df['V1']
                print(f"  保守模式：仅添加 Amount_V1_interaction")
            else:
                # 完整模式：添加Amount与前5个V特征的交互
                for i, v_feat in enumerate(v_features[:5]):
                    df_enhanced[f'Amount_{v_feat}_interaction'] = df['Amount'] * df[v_feat]
        
        print(f"特征工程完成，新增 {len(df_enhanced.columns) - len(df.columns)} 个特征")
        print(f"总特征数: {len(df_enhanced.columns)}")
        
        return df_enhanced

    def get_feature_importance_data(self, X):
        """
        获取特征重要性分析数据
        
        参数:
            X (pd.DataFrame): 特征矩阵
            
        返回:
            pd.DataFrame: 特征统计信息
        """
        feature_stats = []
        
        for col in X.columns:
            stats = {
                'Feature': col,
                'Mean': X[col].mean(),
                'Std': X[col].std(),
                'Min': X[col].min(),
                'Max': X[col].max(),
                'Missing_Count': X[col].isnull().sum(),
                'Missing_Percentage': (X[col].isnull().sum() / len(X)) * 100,
                'Unique_Count': X[col].nunique(),
                'Data_Type': str(X[col].dtype)
            }
            feature_stats.append(stats)
        
        return pd.DataFrame(feature_stats)
