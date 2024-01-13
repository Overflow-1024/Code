import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression

# in_csv_path = "D:/毕设数据/第一层/feature_select_abs/ISall_select2.csv"
#
# df = pd.read_csv(in_csv_path, engine='python', encoding='gb2312')
# df_key = df.loc[:, ['name', 'pod']]
# df_attr = df.drop(['name', 'pod'], axis=1, inplace=False)
# data_X = df_attr.values
# data_y = df['pod'].values
# attr_name = df_attr.columns.values.tolist()
#
# label = df['pod']
# X = df["pcm_fftMag_spectralEntropy_sma_de_lpgain_11"]
# Y = df["pcm_fftMag_spectralEntropy_sma_de_amean_11"]
# X0 = []
# Y0 = []
# X1 = []
# Y1 = []
# for i in range(len(label)):
#     if label[i] == 0:
#         X0.append(X[i])
#         Y0.append(Y[i])
#     else:
#         X1.append(X[i])
#         Y1.append(Y[i])
#
# plt.scatter(X0, Y0, color='red', marker='x')
#
# plt.show()
#
#
# X = data_X
# y = data_y
#
# count1 = 0
# count0 = 0
# for label in y:
#     if label == 0:
#         count0 = count0 + 1
#     elif label == 1:
#         count1 = count1 + 1
#
# print("count0  "+ str(count0))
# print("count1  "+ str(count1))
#
# samples_num = len(X)
# fold = 5
# batch_num = samples_num // fold
#
# # 定义分类器
# RFclf = RandomForestClassifier(n_estimators=10, max_depth=10, min_samples_split=5)
#
# RFacc_list = []
#
# # 划分训练数据和测试数据，进行交叉验证
#
# for i in range(fold):
#     start = i * batch_num
#     end = start + batch_num
#     X_test = X[start:end]
#     y_test = y[start:end]
#     X_train2 = X[end:]
#     y_train2 = y[end:]
#     X_train = X_train2
#     y_train = y_train2
#     if start > 0:
#         X_train1 = X[0:start]
#         y_train1 = y[0:start]
#         X_train = np.concatenate((X_train1, X_train2), axis=0)
#         y_train = np.concatenate((y_train1, y_train2), axis=0)
#
#     # 随机森林
#     RFclf.fit(X_train, y_train)
#     RFacc = RFclf.score(X_test, y_test)
#     y_predict = RFclf.predict(X_test)
#
#     true = 0
#     false = 0
#
#     count = 0
#     for i in range(len(y_test)):
#         if y_predict[i] == 1:
#             count = count + 1
#         if y_predict[i] == y_test[i]:
#             true = true + 1
#         else:
#             false = false + 1
#
#     print("predict_false: " + str(count))
#     RFacc_me = true / (true + false)
#     print(RFacc_me)
#     print(RFacc)
#     RFacc_list.append(RFacc)
#
#
# # 计算交叉验证得到的平均正确率
# RF_MeanAccuracy = np.mean(RFacc_list)
#
#
# print("mean accuracy of RandomForest: " + str(RF_MeanAccuracy))



