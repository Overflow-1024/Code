from sklearn.datasets import load_iris
import numpy as np
from sklearn import preprocessing
from sklearn import feature_selection

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import KBinsDiscretizer

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from scipy.stats import pearsonr
from minepy import MINE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



iris = load_iris()

# Z-score标准化
result = StandardScaler().fit_transform(iris.data)
print(result)

# Min-max Scaling
result = MinMaxScaler().fit_transform(iris.data)
print(result)

# Mean Scaling
result = MaxAbsScaler().fit_transform(iris.data)
print(result)

# 二值化
result = Binarizer(threshold=3).fit_transform(iris.data)
print(result)

# kbins
result = KBinsDiscretizer(iris.data)
print(result)

# 卡方检验
result = SelectKBest(chi2, k=2).fit_transform(iris.data, iris.target)
print(result)

# 相关系数
result = SelectKBest(lambda X, Y: list(np.array(list(map(lambda x: pearsonr(x, Y), X.T))).T), k=2).fit_transform(iris.data, iris.target)   # 相关系数
print(result)


def mic(x, y):
    m = MINE()
    m.compute_score(x, y)
    return (m.mic(), 0.5)


# 互信息
result = SelectKBest(lambda X, Y: list(np.array(list(map(lambda x: mic(x, Y), X.T))).T), k=2).fit_transform(iris.data, iris.target)
print(result)

# Wrapper——基于逻辑回归模型
result = RFE(estimator=LogisticRegression(solver='liblinear', multi_class='auto'), n_features_to_select=2).fit_transform(iris.data, iris.target)
print(result)

# Embedded——基于L1正则化
result = SelectFromModel(LogisticRegression(penalty="l1", C=0.1, solver='liblinear', multi_class='auto')).fit_transform(iris.data, iris.target)
print(result)

# Embedded——基于树模型
result = SelectFromModel(GradientBoostingClassifier()).fit_transform(iris.data, iris.target)
print(result)

# PCA
result = PCA(n_components=2).fit_transform(iris.data)
print(result)

# LDA
result = LinearDiscriminantAnalysis(n_components=2).fit_transform(iris.data, iris.target)
print(result)
