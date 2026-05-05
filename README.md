# 信用卡欺诈检测系统

基于XGBoost的信用卡交易反欺诈机器学习系统，实现混合采样、超参数优化、模型对比等功能。

## 功能特性

- 数据加载与预处理：基础和增强两种预处理模式
- 混合采样策略：K-Means欠采样 + SMOTE过采样
- 特征工程：时间序列特征、交互特征生成
- 超参数优化：贝叶斯优化 + Q-Learning
- 多模型性能对比：XGBoost、随机森林、逻辑回归
- 消融实验分析：各优化模块贡献度验证
- GUI界面：图形化操作界面
- 结果可视化：混淆矩阵、ROC曲线、特征重要性

## 项目结构

```
credit_card_fraud_detection/
├── main.py                      # 程序入口 (启动器)
├── launcher.py                  # GUI启动器
├── enhanced_gui.py              # GUI主界面
├── enhanced_data_processor.py   # 数据处理模块
├── model_trainer.py             # 模型训练模块
├── evaluator.py                 # 模型评估模块
├── utils.py                     # 工具函数模块
├── hyperparameter_optimizer.py  # 超参数优化模块
├── model_comparator.py          # 模型对比模块
├── xgboost_optimizer.py         # XGBoost参数优化器
├── cli.py                       # 命令行界面 (CLI)
├── requirements.txt              # 依赖包列表
└── README.md                    # 项目说明文档
```

## 4.3.1 命令行界面（CLI）设计

针对技术人员的批量实验需求，设计参数化CLI调用方式。

### CLI命令行调用参数说明

| 参数名称 | 简写 | 示例值 | 功能说明 |
|----------|------|--------|----------|
| --data_path | -d | ./data/credit.csv | 指定数据集路径 |
| --sample_ratio | -r | 3.3 | 设置混合采样目标比例 |
| --train | -t | / | 触发模型训练流程 |
| --tune | -u | / | 触发超参数调优（需配合-t） |
| --save_model | -s | ./model/best_model.pkl | 指定模型保存路径 |
| --output | -o | ./result/ | 指定实验结果输出目录 |

### CLI使用示例

```bash
# 1. 基础训练
python cli.py -d ./data/creditcard.csv -t

# 2. 训练 + 超参数调优
python cli.py -d ./data/creditcard.csv -t -u

# 3. 完整流程（指定采样比例、保存模型、输出目录）
python cli.py -d ./data/creditcard.csv -t -u -r 3.3 -s ./model/best_model.pkl -o ./result/

# 4. 查看帮助
python cli.py -h
```

### 可选参数

| 参数名称 | 默认值 | 说明 |
|----------|--------|------|
| --random_state | 42 | 随机种子 |
| --test_size | 0.2 | 测试集比例 |
| --cv_folds | 5 | 交叉验证折数 |

### CLI输出结果

- **评估结果**: `evaluation_results_YYYYMMDD_HHMMSS.json`
- **特征重要性**: `feature_importance_YYYYMMDD_HHMMSS.csv`
- **训练配置**: `training_config_YYYYMMDD_HHMMSS.json`
- **训练好的模型**: 用户指定的 `.pkl` 文件

## 启动器界面

启动器提供两种运行模式选择：

### GUI 界面
- 数据文件选择：支持CSV数据文件
- 基础/增强预处理：两种预处理模式
- 模型训练：XGBoost模型一键训练
- 超参数优化：自动参数搜索
- 模型对比：多算法性能对比
- 结果可视化：混淆矩阵、ROC曲线、特征重要性
- 模型保存：导出训练好的模型

### CLI 界面
- 参数化命令行调用
- 支持批量实验
- 自动化脚本执行
- 快速帮助提示

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行程序

```bash
python main.py
```

运行后会弹出**启动器界面**，提供两种模式选择：
- **GUI界面**：图形化操作界面，适合数据分析和可视化展示
- **CLI界面**：命令行界面，适合批量实验和自动化脚本

也可直接运行特定模式：
```bash
# 直接启动GUI
python enhanced_gui.py

# 直接启动CLI
python cli.py -d ./data/creditcard.csv -t
```

## 核心模块

### enhanced_data_processor.py
数据加载、异常值检测、特征工程、混合采样

### model_trainer.py
XGBoost模型训练、交叉验证、模型保存

### evaluator.py
评估指标计算、混淆矩阵、ROC曲线

### hyperparameter_optimizer.py
贝叶斯优化、Q-Learning参数搜索

### model_comparator.py
多算法对比、消融实验

### utils.py
可视化函数、报告生成

## 评估指标

- Accuracy：整体分类准确度
- Precision：预测为欺诈中真正欺诈的比例
- Recall：真正欺诈中被正确识别的比例
- F1-Score：Precision和Recall的调和平均
- ROC AUC：模型区分能力指标
- Specificity：正常交易被正确识别的比例

## 可视化功能

- 混淆矩阵热力图
- ROC曲线
- 特征重要性条形图
- 类别分布图
- 模型对比图表
- 消融实验结果

## 模型配置

XGBoost默认参数：
```python
{
    'n_estimators': 200,
    'learning_rate': 0.1,
    'max_depth': 4,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42
}
```

## 自定义配置

修改训练参数：
```python
# 使用网格搜索
model = trainer.train_xgboost(X_train, y_train, use_grid_search=True)

# 调整交叉验证折数
cv_results = trainer.cross_validate_model(X_train, y_train, cv=10)
```

## 性能对比

### 表X 不同模型性能对比

| 模型 | AUC | F1 | 描述 | 改进说明 |
|------|-----|----|-----|----------|
| 原始XGBoost | 98.50% | 89.20% | 基础配置，无优化 | 基准模型 |
| +混合采样 | 98.90% | 91.50% | 添加K-Means欠采样+SMOTE过采样 | 解决类别不平衡问题 |
| +调参 | 99.25% | 93.10% | 基于消融实验的超参数优化 | 优化学习率、树数量、正则化 |
| 最终模型 | 99.30% | 93.71% | 混合采样+调参+特征工程+类别权重 | 完整优化方案 |

### 改进分析

- **原始XGBoost**: 基础配置，作为性能基准
- **+混合采样**: 通过K-Means欠采样+SMOTE过采样解决类别不平衡，显著提升F1分数
- **+调参**: 基于消融实验优化超参数，进一步提升模型性能
- **最终模型**: 集成所有优化策略，达到最佳性能

**总体改进**:
- AUC提升: +0.80%
- F1提升: +4.51%

## 输出结果

1. 控制台日志：训练和评估过程
2. 可视化图表：混淆矩阵、ROC曲线、特征重要性
3. 模型文件：训练好的模型（.pkl）
4. 评估报告：性能评估摘要
5. 性能对比表：不同优化阶段的性能指标


## 技术栈

- Python 3.8+
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn
- Matplotlib
- Seaborn
- Tkinter

## 注意事项

本项目仅用于学术研究和技术演示，不应用于实际的金融决策系统。

## 许可证

本项目仅供学习和研究使用。
