import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression


def evaluate(y_predict, y_test):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y_test)):
        if y_predict[i] == 1:
            if y_test[i] == 1:
                TP = TP + 1
            else:
                FP = FP + 1
        if y_predict[i] == 0:
            if y_test[i] == 0:
                TN = TN + 1
            else:
                FN = FN + 1

    confusion_matrix = np.array([[  TP,      FP,         TP + FP],
                                 [  FN,      TN,         TN + FN],
                                 [TP + FN, FP + TN, TP + FP + TN + FN]])

    mean_accuracy = (TP + TN) / (TP + FP + TN + FN)
    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)

    if TP + FN == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)

    if precision + recall == 0:
        Fscore = 0
    else:
        Fscore = precision * recall * 2 / (precision + recall)

    return mean_accuracy, confusion_matrix, precision, recall, Fscore


def train_one_model(X, y, model):

    y_list = list(y)
    positive = y_list.count(1)
    negative = y_list.count(0)

    proportion = positive / (positive + negative)

    # 定义分类器
    if model == 'DT':
        clf = DecisionTreeClassifier(splitter='best', max_depth=50)
        model_name = "DecisionTree"
    elif model == 'SVM':
        clf = svm.SVC(kernel='rbf', gamma='scale', decision_function_shape='ovr', class_weight='balanced')
        model_name = "SupportVectorMachine"
    elif model == 'LR':
        clf = LogisticRegression(penalty='l2', solver='liblinear', multi_class='ovr')
        model_name = "LogisticRegression"
    elif model == 'OCSVM':
        clf = svm.OneClassSVM(kernel='rbf', gamma='scale', nu=proportion)
        model_name = "OneClassSVM"
    else:
        clf = RandomForestClassifier(n_estimators=100, max_depth=50)
        model_name = "RandomForest"

    samples_num = len(X)
    fold = 5
    batch_num = samples_num // fold

    y_test_all = np.zeros((fold, batch_num), dtype=int)
    y_predict_all = np.zeros((fold, batch_num), dtype=int)

    # 划分训练数据和测试数据，进行交叉验证
    for i in range(fold):
        start = i * batch_num
        end = start + batch_num
        X_test = X[start:end]
        y_test = y[start:end]
        if start == 0:
            X_train = X[end:]
            y_train = y[end:]
        elif end == X.shape[0]:
            X_train = X[0:start]
            y_train = y[0:start]
        else:
            X_train = np.concatenate((X[0:start], X[end:]), axis=0)
            y_train = np.concatenate((y[0:start], y[end:]), axis=0)

        # 训练分类器
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        if model == 'OCSVM':
            y_predict = np.where(y_predict > 0, 0, 1)

        y_test_all[i] = y_test
        y_predict_all[i] = y_predict

    # 计算混淆矩阵，精确率，召回率，F值，平均正确率
    mean_accuracy, confusion_matrix, precision, recall, Fscore = evaluate(y_predict_all.flatten(), y_test_all.flatten())
    print("mean accuracy of " + model_name + ": " + str(mean_accuracy))
    print("confusion matrix of " + model_name + ": ")
    print(confusion_matrix)
    print("precision of " + model_name + ": " + str(precision))
    print("recall of " + model_name + ": " + str(recall))
    print("Fscore of " + model_name + ": " + str(Fscore))

    return mean_accuracy, Fscore


def train_model(X, y):

    # 使用不同的分类器
    DT_mean_accuracy, DT_Fscore = train_one_model(X, y, 'DT')
    RF_mean_accuracy, RF_Fscore = train_one_model(X, y, 'RF')
    SVM_mean_accuracy, SVM_Fscore = train_one_model(X, y, 'SVM')
    LR_mean_accuracy, LR_Fscore = train_one_model(X, y, 'LR')

    return DT_Fscore, RF_Fscore, SVM_Fscore, LR_Fscore



