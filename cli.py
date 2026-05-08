#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信用卡欺诈检测系统 - 命令行界面(CLI)
针对技术人员的批量实验需求，提供参数化CLI调用方式
"""

import argparse
import os
import sys
import json
import pickle
from datetime import datetime

import pandas as pd
import numpy as np

from enhanced_data_processor import EnhancedDataProcessor
from model_trainer import ModelTrainer
from hyperparameter_optimizer import HybridOptimizer, BayesianOptimizer
from evaluator import ModelEvaluator


def create_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description='信用卡欺诈检测系统 - 命令行工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基础训练
  python cli.py -d ./data/credit.csv -t -s ./model/best_model.pkl
  
  # 训练+超参数调优
  python cli.py -d ./data/credit.csv -t -u -s ./model/best_model.pkl
  
  # 指定采样比例和输出目录
  python cli.py -d ./data/credit.csv -t -r 3.3 -o ./result/ -s ./model/model.pkl
        """
    )
    
    # 数据路径参数
    parser.add_argument(
        '-d', '--data_path',
        type=str,
        required=True,
        help='指定数据集路径 (示例: ../creditcard.csv)'
    )
    
    # 混合采样目标比例
    parser.add_argument(
        '-r', '--sample_ratio',
        type=float,
        default=5.0,
        help='设置混合采样目标比例 (默认: 5.0)'
    )
    
    # 触发模型训练
    parser.add_argument(
        '-t', '--train',
        action='store_true',
        help='触发模型训练流程'
    )
    
    # 触发超参数调优
    parser.add_argument(
        '-u', '--tune',
        action='store_true',
        help='触发超参数调优 (需配合 -t 使用)'
    )
    
    # 模型保存路径
    parser.add_argument(
        '-s', '--save_model',
        type=str,
        default=None,
        help='指定模型保存路径 (示例: ./model/best_model.pkl)'
    )
    
    # 输出目录
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='./result/',
        help='指定实验结果输出目录 (默认: ./result/)'
    )
    
    # 随机种子
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='设置随机种子 (默认: 42)'
    )
    
    # 测试集比例
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.2,
        help='设置测试集比例 (默认: 0.2)'
    )
    
    # 交叉验证折数
    parser.add_argument(
        '--cv_folds',
        type=int,
        default=5,
        help='设置交叉验证折数 (默认: 5)'
    )
    
    return parser


def print_config(args):
    """打印配置信息"""
    print("=" * 60)
    print("信用卡欺诈检测系统 - CLI配置")
    print("=" * 60)
    print(f"数据集路径: {args.data_path}")
    print(f"混合采样比例: {args.sample_ratio}")
    print(f"训练模式: {'开启' if args.train else '关闭'}")
    print(f"超参数调优: {'开启' if args.tune else '关闭'}")
    print(f"模型保存路径: {args.save_model if args.save_model else '未指定'}")
    print(f"结果输出目录: {args.output}")
    print(f"随机种子: {args.random_state}")
    print(f"测试集比例: {args.test_size}")
    print(f"交叉验证折数: {args.cv_folds}")
    print("=" * 60)


def ensure_dir(directory):
    """确保目录存在"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"创建目录: {directory}")


def save_results(results, output_dir, filename):
    """保存评估结果到JSON文件"""
    # 过滤不可序列化的数据
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, (int, float, str, bool, list, dict, tuple)):
            try:
                json.dumps({key: value})
                serializable_results[key] = value
            except (TypeError, ValueError):
                pass
    
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    print(f"评估结果已保存: {filepath}")
    return filepath


def save_feature_importance(feature_importance_df, output_dir, filename):
    """保存特征重要性到CSV文件"""
    filepath = os.path.join(output_dir, filename)
    feature_importance_df.to_csv(filepath, index=False, encoding='utf-8')
    print(f"特征重要性已保存: {filepath}")
    return filepath


def run_training_pipeline(args):
    """执行训练流程"""
    print("\n" + "=" * 60)
    print("开始训练流程")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n[1/6] 加载数据...")
    processor = EnhancedDataProcessor()
    df = processor.load_data(args.data_path)
    
    if df is None:
        print("错误: 数据加载失败，请检查数据路径")
        sys.exit(1)
    
    # 显示数据基本信息
    print(f"数据集形状: {df.shape}")
    if 'Class' in df.columns:
        fraud_count = (df['Class'] == 1).sum()
        normal_count = (df['Class'] == 0).sum()
        print(f"欺诈样本数: {fraud_count}")
        print(f"正常样本数: {normal_count}")
        print(f"欺诈比例: {fraud_count / len(df) * 100:.4f}%")
    
    # 2. 数据预处理
    print(f"\n[2/6] 数据预处理 (采样比例: {args.sample_ratio})...")
    X_train, X_test, y_train, y_test = processor.preprocess_data_split(
        df,
        test_size=args.test_size,
        use_hybrid_sampling=True,
        use_advanced_features=True,
        target_ratio=int(args.sample_ratio),
        random_state=args.random_state
    )
    
    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    
    # 3. 超参数调优（如果启用）
    trainer = ModelTrainer()
    best_params = None
    
    if args.tune:
        print(f"\n[3/6] 执行超参数调优 (贝叶斯优化)...")
        optimizer = BayesianOptimizer(
            max_iter=20,
            cv_folds=args.cv_folds,
            random_state=args.random_state
        )
        best_params = optimizer.optimize(X_train, y_train, verbose=True)
        print("\n最佳参数:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
    else:
        print("\n[3/6] 跳过超参数调优，使用默认优化参数")
    
    # 4. 模型训练
    print(f"\n[4/6] 模型训练...")
    if best_params:
        model = trainer.train_xgboost(X_train, y_train, **best_params)
    else:
        model = trainer.train_xgboost(X_train, y_train)
    
    # 5. 模型评估
    print(f"\n[5/6] 模型评估...")
    evaluator = ModelEvaluator()
    
    # 预测
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 综合评估
    results = evaluator.comprehensive_evaluation(y_test, y_pred, y_pred_proba)
    evaluator.print_evaluation_summary(results)
    
    # 特征重要性
    feature_names = X_train.columns.tolist()
    feature_importance_df = evaluator.get_feature_importance(model, feature_names)
    if feature_importance_df is not None:
        print("\nTop 10 重要特征:")
        print(feature_importance_df.head(10).to_string(index=False))
    
    # 交叉验证
    print(f"\n执行 {args.cv_folds} 折交叉验证...")
    cv_results = trainer.cross_validate_model(X_train, y_train, cv=args.cv_folds)
    
    # 6. 保存结果
    print(f"\n[6/6] 保存结果...")
    
    # 确保输出目录存在
    ensure_dir(args.output)
    
    # 生成时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存评估结果
    results_filename = f'evaluation_results_{timestamp}.json'
    save_results(results, args.output, results_filename)
    
    # 保存特征重要性
    if feature_importance_df is not None:
        fi_filename = f'feature_importance_{timestamp}.csv'
        save_feature_importance(feature_importance_df, args.output, fi_filename)
    
    # 保存模型
    if args.save_model:
        # 确保模型保存目录存在
        model_dir = os.path.dirname(args.save_model)
        if model_dir:
            ensure_dir(model_dir)
        trainer.save_model(model, args.save_model)
    
    # 保存训练配置
    config = {
        'data_path': args.data_path,
        'sample_ratio': args.sample_ratio,
        'random_state': args.random_state,
        'test_size': args.test_size,
        'cv_folds': args.cv_folds,
        'use_tuning': args.tune,
        'best_params': best_params,
        'timestamp': timestamp,
        'evaluation_results': {
            'accuracy': results.get('accuracy', 0),
            'precision': results.get('precision', 0),
            'recall': results.get('recall', 0),
            'f1_score': results.get('f1_score', 0),
            'roc_auc': results.get('roc_auc', 0)
        }
    }
    
    config_filename = f'training_config_{timestamp}.json'
    config_path = os.path.join(args.output, config_filename)
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"训练配置已保存: {config_path}")
    
    print("\n" + "=" * 60)
    print("训练流程完成！")
    print("=" * 60)
    
    return results


def main():
    """主函数"""
    # 创建参数解析器
    parser = create_parser()
    
    # 解析参数
    args = parser.parse_args()
    
    # 打印配置
    print_config(args)
    
    # 检查必要参数
    if not args.train:
        print("\n错误: 必须使用 -t 或 --train 参数触发训练流程")
        parser.print_help()
        sys.exit(1)
    
    # 检查数据文件是否存在
    if not os.path.exists(args.data_path):
        print(f"\n错误: 数据文件不存在: {args.data_path}")
        sys.exit(1)
    
    # 执行训练流程
    try:
        results = run_training_pipeline(args)
        
        # 返回最终性能指标
        print(f"\n最终性能指标:")
        print(f"  F1-Score: {results.get('f1_score', 0):.6f}")
        print(f"  ROC-AUC: {results.get('roc_auc', 0):.6f}")
        print(f"  Recall: {results.get('recall', 0):.6f}")
        print(f"  Precision: {results.get('precision', 0):.6f}")
        
        sys.exit(0)
        
    except Exception as e:
        print(f"\n错误: 训练过程中发生异常: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
