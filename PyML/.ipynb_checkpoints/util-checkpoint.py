from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_decision_regions(X, y, classifier, resolution=0.02, test_idx=None):
    markers = ("s", "x", "o", "^", "v")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    # 根据y有多少个类别初始化多少个color。生成colormap对象
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    # 获取到x_1 x_2 的取值范围，便于绘制坐标轴。
    x1_min, x1_max = X[:, 0].min() -1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() -1, X[:, 1].max() + 1
    
    # 生成二维坐标矩阵，方便绘制
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    
    # 为二维坐标矩阵预测每一点的预测结果
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    # 绘制等高线
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    
    # 指定坐标轴范围
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # 绘制样本点
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)
    
    # hightlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c="", alpha=1.0,edgecolors='b', linewidth=1, marker="o", s=66, label="test set")

def load_iris_data():
    df = pd.read_csv("iris.data", header = None)

    # 取出数据中的分类信息
    y = df.iloc[0:100, 4].values
    # 讲这些信息用离散化数据表示
    y = np.where(y == "Iris-setosa", -1, 1)

    # 取出我们需要的特征信息
    X = df.iloc[0:100, [0, 2]].values
    return X, y

def load_std_data():
    from sklearn import datasets
    from sklearn.preprocessing import StandardScaler
    from sklearn.cross_validation import train_test_split
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y,           test_size=0.3, random_state = 0)
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    return X_train_std, y_train, X_test_std, y_test
    
    