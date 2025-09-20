一、 机器学习基本概念 (Conceptual Knowledge)
  1.问题定性 (Problem Framing):
      监督学习 (Supervised Learning): 我们的数据集包含特征（花瓣/萼片尺寸）和与之对应的标记（鸢尾花品种），因此属于监督学习。
      分类任务 (Classification Task): 预测的目标是离散的类别（三个品种之一），而非连续的数值，因此是分类任务。
      多分类 (Multi-class Classification): 因为类别数量大于两个（共三个），所以是多分类任务。
      泛化能力 (Generalization):机器学习的核心目标是让模型在新颖、未见过的数据上表现良好，而不仅仅是在训练数据上。为了评估泛化能力，必须将数据集划分为训练集和测试集。模型在训练阶段绝对不能接触测试集数据。
  2.数据核心术语:
      特征 (Features): 用于预测的输入变量。在本项目中是4个，存储在变量 X 中，其形状为 (样本数, 特征数)，即 (150, 4)。
      标记 (Labels / Target): 需要预测的目标变量。在本项目中是花的品种，存储在变量 y 中，其形状为 (样本数,)，即 (150,)。
      样本 (Sample): 数据集中的一条记录，即一朵花的所有信息。
      
二、 scikit-learn 核心使用方法 (Practical Skills)
  1.标准化的API接口 (Consistent API):
      数据加载: from sklearn.datasets import load_...
      模型选择: from sklearn.module import ModelName
      训练模型: model.fit(X_train, y_train)
      预测结果: y_pred = model.predict(X_test)
      评估性能: from sklearn.metrics import metric_function
  2.关键模块与函数:
      sklearn.datasets: 用于加载内置的标准数据集。
      load_iris(): 加载鸢尾花数据集。返回一个 Bunch 对象，包含 .data, .target, .feature_names, .target_names 等属性。
      sklearn.model_selection: 用于数据划分和模型选择。
      train_test_split(X, y, test_size=..., random_state=...): 核心函数，用于将数据集随机划分为训练集和测试集。
      test_size: 指定测试集所占的比例（例如 0.3 代表30%）。
      random_state: 设置一个固定的整数，可以确保每次划分的结果都完全相同，便于代码复现。
      sklearn.neighbors: 包含基于邻居的算法。
      KNeighborsClassifier(n_neighbors=...): K-近邻分类器。
      n_neighbors: 核心超参数，指定在做预测时参考最近的邻居数量（本项目中为3）。
      sklearn.metrics: 包含用于评估模型性能的各种指标。
      accuracy_score(y_true, y_pred): 计算分类准确率。y_true 是真实标记，y_pred 是模型预测的标记。
      
三、 完整的机器学习项目流程 (The Workflow)
可以记在心里，并应用到未来几乎所有监督学习项目中的标准流程：
  1.导入库 (Import Libraries):
      from sklearn.datasets import load_iris
      from sklearn.model_selection import train_test_split
      from sklearn.neighbors import KNeighborsClassifier
      from sklearn.metrics import accuracy_score
  2.加载并准备数据 (Load & Prepare Data):
      dataset = load_iris()
      X = dataset.data
      y = dataset.target
  3.划分数据 (Split Data):
      X_train, X_test, y_train, y_test = train_test_split(X, y, ...)
  4.初始化模型 (Initialize Model):
      model = KNeighborsClassifier(n_neighbors=3)
  5.训练模型 (Train Model):
      model.fit(X_train, y_train)
  6.进行预测 (Make Predictions):
      y_pred = model.predict(X_test)
  7.评估模型 (Evaluate Model):
      accuracy = accuracy_score(y_test, y_pred)
      print(f"Accuracy: {accuracy}")
