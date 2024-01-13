import numpy as np
import pandas as pd
from minepy import MINE
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def mic(x, y):
    m = MINE()
    m.compute_score(x, y)
    return m.mic(), 0.5


def fisher_ratio(x, y):
    positive_sample = []
    negative_sample = []

    for i in range(len(y)):
        if y[i] == 1:
            positive_sample.append(x[i])
        else:
            negative_sample.append(x[i])

    positive = np.array(positive_sample)
    negative = np.array(negative_sample)

    fisher_score = (positive.mean() - negative.mean()) * (positive.mean() - negative.mean()) / (positive.var() + negative.var())

    return fisher_score


# 利用sklearn提供的SelectKBset进行特征选择
# 得到的attr_index按照其在原矩阵中的列数从小到大排序
def select_feature(X, y, feature_num, score_function):

    print("before selection: ")
    print(X.shape)

    X = VarianceThreshold(threshold=0).fit_transform(X)

    if score_function == 'mic':
        selector = SelectKBest(lambda X, Y: list(np.array(list(map(lambda x: mic(x, Y), X.T))).T), k=feature_num)
    else:
        selector = SelectKBest(lambda X, Y: list(np.array(list(map(lambda x: pearsonr(x, Y), X.T))).T), k=feature_num)

    X_select = selector.fit_transform(X, y)
    attr_index = selector.get_support(indices=True)

    print("after selection: ")
    print(X_select.shape)

    return X_select, y, attr_index


# 自己手写的特征选择过程
# 得到的attr_index按照评分由高到低排序
def select_feature_detail(X, y, feature_num, score_function, abs_flag):

    print("before selection: ")
    print(X.shape)

    X = VarianceThreshold(threshold=0).fit_transform(X)

    attr_dict = {}
    attr_index = []
    for i in range(X.shape[1]):
        print(i)

        if score_function == 'pearson':
            score, p = pearsonr(X[:, i], y)
            if abs_flag:
                score = abs(score)
        elif score_function == 'mic':
            score, p = mic(X[:, i], y)
        else:
            score = fisher_ratio(X[:, i], y)

        attr_dict[i] = score

    attr_seq = sorted(attr_dict.items(), key=lambda item: item[1], reverse=True)

    if feature_num == 256:
        score = []
        for i in range(len(attr_seq)):
            score.append(attr_seq[i][1])

        # 画图
        plt.plot(range(len(score)), score, color='red', linestyle="-", marker=".", linewidth=1)

        plt.xlabel("feature")
        plt.ylabel("score")

        plt.savefig("score_picture.png")
        plt.show()


    for i in range(feature_num):
        attr_index.append(attr_seq[i][0])

    X_select = X[:, attr_index]

    print("after selection: ")
    print(X_select.shape)

    return X_select, y, attr_index


# 计算每个特征的评分
def get_feature_score(X, y, score_function, abs_flag):

    X = VarianceThreshold(threshold=0).fit_transform(X)

    attr_score = []

    for i in range(X.shape[1]):

        if score_function == 'pearson':
            score, p = pearsonr(X[:, i], y)
            if abs_flag:
                score = abs(score)
        elif score_function == 'mic':
            score, p = mic(X[:, i], y)
        else:
            score = fisher_ratio(X[:, i], y)

        attr_score.append(score)

    return attr_score


# 计算每个特征的评分并排序
def get_feature_score_sorted(X, y, score_function, abs_flag):

    X = VarianceThreshold(threshold=0).fit_transform(X)

    attr_dict = {}

    for i in range(X.shape[1]):

        print(i)
        if score_function == 'pearson':
            score, p = pearsonr(X[:, i], y)
            if abs_flag:
                score = abs(score)
        elif score_function == 'mic':
            score, p = mic(X[:, i], y)
        else:
            score = fisher_ratio(X[:, i], y)

        attr_dict[i] = score

    attr_seq = sorted(attr_dict.items(), key=lambda item: item[1], reverse=True)

    return attr_seq


def PCA_dimensionality_reduction(X, y, components_num):

    selector = PCA(n_components=components_num)
    X_select = selector.fit_transform(X)
    ratio = selector.explained_variance_ratio_

    plt.bar(range(len(ratio)), ratio, color='blue')

    plt.xlabel("dimension")
    plt.ylabel("contribution")

    plt.show()
    return X_select, y, ratio



