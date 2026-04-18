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
├── main.py                      # 程序入口
├── enhanced_gui.py              # GUI界面
├── enhanced_data_processor.py   # 数据处理模块
├── model_trainer.py             # 模型训练模块
├── evaluator.py                 # 模型评估模块
├── utils.py                     # 工具函数模块
├── hyperparameter_optimizer.py  # 超参数优化模块
├── model_comparator.py          # 模型对比模块
├── xgboost_optimizer.py         # XGBoost参数优化器
├── requirements.txt              # 依赖包列表
└── README.md                    # 项目说明文档
```

## GUI界面功能

- 数据文件选择：支持CSV数据文件
- 基础/增强预处理：两种预处理模式
- 模型训练：XGBoost模型一键训练
- 超参数优化：自动参数搜索
- 模型对比：多算法性能对比
- 结果可视化：混淆矩阵、ROC曲线、特征重要性
- 模型保存：导出训练好的模型

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行程序

```bash
# GUI版本
python main.py
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

## 输出结果

1. 控制台日志：训练和评估过程
2. 可视化图表：混淆矩阵、ROC曲线、特征重要性
3. 模型文件：训练好的模型（.pkl）
4. 评估报告：性能评估摘要


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
