# 信用卡欺诈检测系统 - 增强版

基于XGBoost的信用卡交易反欺诈机器学习系统，实现开题报告要求的混合采样、超参数优化、模型对比等高级功能。

## 🎯 功能特性

### 核心功能
- � **数据加载与预处理**: 基础和增强两种预处理模式
- 🧬 **混合采样策略**: K-Means欠采样 + SMOTE过采样
- 🔧 **高级特征工程**: 滑动窗口统计、时间序列特征、交互特征
- 🤖 **智能超参数优化**: 贝叶斯优化 + Q-Learning动态调整
- 📊 **多模型性能对比**: XGBoost、随机森林、逻辑回归等
- 🔬 **消融实验分析**: 量化各优化模块的贡献度
- 🎨 **双版本GUI界面**: 基础版和增强版图形界面
- 📈 **全面结果可视化**: 混淆矩阵、ROC曲线、特征重要性、对比图表

### 增强功能（符合开题报告要求）
- ✅ **混合采样**: K-Means聚类欠采样 + SMOTE过采样，比例优化至1:5
- ✅ **特征工程**: 交易金额统计、时间序列特征、变异系数、交互特征
- ✅ **超参数优化**: 贝叶斯优化结合Q-Learning算法，500轮自动调优
- ✅ **模型对比**: 五折交叉验证，多算法性能评估
- ✅ **消融实验**: 验证各优化模块的有效性

## 📁 项目结构

```
credit_card_fraud_detection/
├── main.py                      # 命令行主程序（基础版）
├── gui.py                       # 基础版GUI界面
├── enhanced_gui.py               # 增强版GUI界面 ⭐
├── data_processor.py             # 基础数据处理模块
├── enhanced_data_processor.py    # 增强数据处理模块 ⭐
├── model_trainer.py              # 模型训练模块
├── evaluator.py                  # 模型评估模块
├── utils.py                      # 工具函数模块
├── hyperparameter_optimizer.py   # 超参数优化模块 ⭐
├── model_comparator.py           # 模型对比模块 ⭐
├── requirements.txt              # 依赖包列表
├── start_gui.bat                 # 基础版启动脚本
├── start_gui.sh                  # 基础版启动脚本
├── start_enhanced_gui.bat        # 增强版启动脚本 ⭐
├── start_enhanced_gui.sh         # 增强版启动脚本 ⭐
└── README.md                     # 项目说明文档
```

## 🚀 快速开始

### 方法一：使用增强版启动脚本（推荐）

**Windows用户:**
```bash
双击运行 start_enhanced_gui.bat
```

**Linux/Mac用户:**
```bash
chmod +x start_enhanced_gui.sh
./start_enhanced_gui.sh
```

### 方法二：手动安装运行

1. **创建虚拟环境**
```bash
python -m venv .venv
```

2. **激活虚拟环境**
```bash
# Windows
.venv\Scripts\activate.bat

# Linux/Mac
source .venv/bin/activate
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

4. **运行程序**

图形界面版本：
```bash
python gui.py
```

命令行版本：
```bash
python main.py
```

## 📊 核心功能

### 数据处理模块 (`data_processor.py`)
- 数据加载和探索性分析
- 缺失值处理
- 特征缩放和标准化
- 类别分布分析
- 异常值检测

### 模型训练模块 (`model_trainer.py`)
- XGBoost模型训练
- 超参数调优（网格搜索）
- 交叉验证
- 模型保存和加载
- 不平衡数据处理

### 模型评估模块 (`evaluator.py`)
- 多种评估指标计算
- 混淆矩阵分析
- ROC曲线和AUC计算
- 特征重要性分析
- 模型性能比较

### 工具函数模块 (`utils.py`)
- 数据可视化函数
- 结果报告生成
- 文件保存工具
- 图表样式设置

## 📈 评估指标

项目使用多种指标来评估模型性能：

- **准确率 (Accuracy)**: 整体分类准确度
- **精确率 (Precision)**: 预测为欺诈的交易中真正欺诈的比例
- **召回率 (Recall)**: 真正欺诈交易中被正确识别的比例
- **F1分数**: 精确率和召回率的调和平均
- **ROC AUC**: 模型区分能力的综合指标
- **特异性**: 正常交易被正确识别的比例

## 🎨 可视化功能

- **混淆矩阵热力图**: 直观展示分类结果
- **ROC曲线**: 展示模型在不同阈值下的性能
- **特征重要性条形图**: 显示最重要的特征
- **类别分布图**: 展示数据不平衡情况
- **学习曲线**: 分析模型学习过程

## ⚙️ 模型配置

### XGBoost参数
```python
{
    'n_estimators': 200,        # 树的数量
    'learning_rate': 0.1,       # 学习率
    'max_depth': 4,             # 树的最大深度
    'subsample': 0.8,           # 子采样比例
    'colsample_bytree': 0.8,    # 列采样比例
    'scale_pos_weight': auto,   # 类别权重（自动计算）
    'reg_alpha': 0.1,           # L1正则化
    'reg_lambda': 1.0,          # L2正则化
    'random_state': 42
}
```

## 🔧 自定义配置

可以通过修改 `main.py` 中的参数来自定义模型：

```python
# 使用网格搜索调优
model = trainer.train_xgboost(X_train, y_train, use_grid_search=True)

# 调整交叉验证折数
cv_results = trainer.cross_validate_model(X_train, y_train, cv=10)
```

## 📝 输出结果

程序运行后会输出：

1. **控制台日志**: 详细的训练和评估过程
2. **可视化图表**: 混淆矩阵、ROC曲线、特征重要性等
3. **模型文件**: 训练好的模型保存为 `.pkl` 文件
4. **评估报告**: 完整的性能评估摘要

## 🤝 与毕设项目的区别

相比原始的Jupyter Notebook，本项目具有以下优势：

1. **代码结构**: 从单一notebook文件重构为模块化项目
2. **可维护性**: 清晰的模块划分，便于维护和扩展
3. **中文支持**: 完整的中文注释和文档
4. **工程化**: 符合软件工程最佳实践
5. **可复用性**: 各模块可独立使用和测试
6. **部署友好**: 便于打包和部署到生产环境

## 📚 技术栈

- **Python 3.7+**: 主要编程语言
- **Pandas**: 数据处理和分析
- **NumPy**: 数值计算
- **Scikit-learn**: 机器学习工具包
- **XGBoost**: 梯度提升算法
- **Matplotlib/Seaborn**: 数据可视化

## 🐛 常见问题

### 1. 数据集下载问题
- 确保从Kaggle下载完整的数据集
- 文件名应为 `creditcard.csv`
- 检查文件路径是否正确

### 2. 依赖包安装问题
```bash
# 如果遇到依赖冲突，建议使用虚拟环境
python -m venv fraud_detection_env
source fraud_detection_env/bin/activate  # Linux/Mac
# 或
fraud_detection_env\Scripts\activate     # Windows
pip install -r requirements.txt
```

### 3. 内存不足
- 数据集较大，建议至少8GB内存
- 可以减少 `n_estimators` 参数来降低内存使用

## 📄 许可证

本项目仅供学习和研究使用。

## 🙏 致谢

- 感谢Kaggle提供的信用卡欺诈检测数据集
- 感谢XGBoost开发团队提供的优秀算法
- 感谢开源社区的贡献

---

**注意**: 本项目仅用于学术研究和技术演示，不应用于实际的金融决策系统。实际应用中需要更复杂的安全措施和合规要求。
