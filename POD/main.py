import numpy as np
import pandas as pd
import sys
import csv
import preprocessing
import feature_selection
import model
import matplotlib.pyplot as plt


data_path = "D:/毕设数据/第一层/ISall_NN.csv"
valid_data_path = "D:/毕设数据/第一层/ISall_valid.csv"

result_pearson_path = "D:/毕设数据/第一层/feature_select_pearson/"
result_mic_path = "D:/毕设数据/第一层/feature_select_mic/"
result_fisher_path = "D:/毕设数据/第一层/feature_select_fisher/"
file_name = "ISall_select.csv"
score_path = "D:/毕设数据/第一层/feature_score.csv"

result_pearson_picture_path = "result_pearson.png"
result_mic_picture_path = "result_mic.png"
result_fisher_picture_path = "result_fisher.png"
score_picture_path = "feature_score.png"

in_csv_path = data_path
out_csv_path = result_mic_path
picture_path = result_mic_picture_path


def select_feature_and_train_model(score_fun):

    # 读取数据文件
    df = pd.read_csv(in_csv_path, engine='python', encoding='gb2312')
    df_key = df.loc[:, ['name', 'pod']]
    df_attr = df.drop(['name', 'pod'], axis=1, inplace=False)
    data_X = df_attr.values
    data_y = df['pod'].values
    attr_name = df_attr.columns.values.tolist()

    # 过采样，然后洗牌
    data_X, data_y = preprocessing.over_sampling_smote(data_X, data_y)
    data_X, data_y = preprocessing.shuffle(data_X, data_y)

    feature_num_list = [256, 128, 64, 32, 16, 8, 4, 2]

    DT_Fscore_list = []
    RF_Fscore_list = []
    SVM_Fscore_list = []
    LR_Fscore_list = []

    for i in range(len(feature_num_list)):
        print("select feature num: " + str(feature_num_list[i]))

        X, y, attr_index = feature_selection.select_feature_detail(data_X, data_y, feature_num_list[i],
                                                                   score_function=score_fun, abs_flag=True)

        # 输出为文件保存
        df_attr_select = df_attr.iloc[:, attr_index]
        df_select = pd.merge(df_key, df_attr_select, how='inner', on=None, left_index=True, right_index=True)
        str1 = file_name[:-4]
        str2 = file_name[-4:]
        out_csv_path_file = out_csv_path + str1 + str(feature_num_list[i]) + str2
        df_select.to_csv(out_csv_path_file, encoding='gb2312', index=False)

        DT_Fscore, RF_Fscore, SVM_Fscore, LR_Fscore = model.train_model(X, y)
        DT_Fscore_list.append(DT_Fscore)
        RF_Fscore_list.append(RF_Fscore)
        SVM_Fscore_list.append(SVM_Fscore)
        LR_Fscore_list.append(LR_Fscore)

        data_X = X
        df_attr = df_attr_select

    print("Accuracy of DecisionTree: ")
    print(DT_Fscore_list)
    print("Accuracy of RandomForest: ")
    print(RF_Fscore_list)
    print("Accuracy of SupportVectorMachine: ")
    print(SVM_Fscore_list)
    print("Accuracy of LogisticRegression: ")
    print(LR_Fscore_list)

    # 画图
    l1, = plt.plot(feature_num_list, DT_Fscore_list, color='red', linestyle="-", marker="o", linewidth=1)
    l2, = plt.plot(feature_num_list, RF_Fscore_list, color='green', linestyle="-", marker="o", linewidth=1)
    l3, = plt.plot(feature_num_list, SVM_Fscore_list, color='blue', linestyle="-", marker="o", linewidth=1)
    l4, = plt.plot(feature_num_list, LR_Fscore_list, color='yellow', linestyle="-", marker="o", linewidth=1)

    plt.xlabel("feature_num")
    plt.ylabel("Fscore")
    plt.legend(handles=[l1, l2, l3, l4], labels=['DT', 'RF', 'SVM', 'LR'], loc='best')

    plt.savefig(picture_path)
    plt.show()


def get_feature_score(score_fun):

    # 读取数据文件
    df = pd.read_csv(in_csv_path, engine='python', encoding='gb2312')
    df_key = df.loc[:, ['name', 'pod']]
    df_attr = df.drop(['name', 'pod'], axis=1, inplace=False)
    data_X = df_attr.values
    data_y = df['pod'].values
    attr_name = df_attr.columns.values.tolist()

    # 过采样，然后洗牌
    data_X, data_y = preprocessing.over_sampling_smote(data_X, data_y)
    data_X, data_y = preprocessing.shuffle(data_X, data_y)

    attr_seq = feature_selection.get_feature_score_sorted(data_X, data_y, score_function=score_fun, abs_flag=True)
    feature = []
    score = []
    for i in range(len(attr_seq)):
        feature.append(attr_name[attr_seq[i][0]])
        score.append(attr_seq[i][1])

    score_data = {"feature": feature, "score": score}
    df_score = pd.DataFrame(score_data)
    df_score.to_csv(out_csv_path, encoding='gb2312', index=False)

    # 画图
    plt.plot(range(len(attr_seq)), score, color='red', linestyle="-", marker=".", linewidth=1)

    plt.xlabel("feature")
    plt.ylabel("score")

    plt.savefig(picture_path)
    plt.show()


def get_feature_score_overlap():

    # 读取数据文件
    df = pd.read_csv(in_csv_path, engine='python', encoding='gb2312')
    df_key = df.loc[:, ['name', 'pod']]
    df_attr = df.drop(['name', 'pod'], axis=1, inplace=False)
    data_X = df_attr.values
    data_y = df['pod'].values
    attr_name = df_attr.columns.values.tolist()

    # 过采样，然后洗牌
    data_X, data_y = preprocessing.over_sampling_smote(data_X, data_y)
    data_X, data_y = preprocessing.shuffle(data_X, data_y)

    pearson_score = feature_selection.get_feature_score_sorted(data_X, data_y, score_function="pearson", abs_flag=True)
    fisher_score = feature_selection.get_feature_score_sorted(data_X, data_y, score_function="fisher", abs_flag=True)

    pearson_attr = []
    fisher_attr = []

    for i in range(200):
        pearson_attr.append(pearson_score[i][0])
        fisher_attr.append(fisher_score[i][0])

    overlap_attr_index = list(set(pearson_attr).intersection(set(fisher_attr)))
    overlap_attr_index.sort()

    overlap_attr_name = []
    for i in range(len(overlap_attr_index)):
        overlap_attr_name.append(attr_name[overlap_attr_index[i]])

    print(overlap_attr_index)
    print(overlap_attr_name)

    score_data = {"attr_index": overlap_attr_index, "attr_name": overlap_attr_name}
    df_score = pd.DataFrame(score_data)
    df_score.to_csv(out_csv_path, encoding='gb2312', index=False)


def PCA(score_fun):

    contribution_list = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

    # 读取数据文件
    df = pd.read_csv(in_csv_path, engine='python', encoding='gb2312')
    df_key = df.loc[:, ['name', 'pod']]
    df_attr = df.drop(['name', 'pod'], axis=1, inplace=False)
    data_X = df_attr.values
    data_y = df['pod'].values
    attr_name = df_attr.columns.values.tolist()

    # 过采样，然后洗牌
    data_X, data_y = preprocessing.over_sampling_smote(data_X, data_y)
    data_X, data_y = preprocessing.shuffle(data_X, data_y)

    # 选出前4000个特征做PCA
    X, y, attr_index = feature_selection.select_feature_detail(data_X, data_y, 4000, score_function=score_fun,
                                                               abs_flag=True)

    # PCA
    X_new, y_new, ratio = feature_selection.PCA_dimensionality_reduction(X, y, components_num=100)
    ratio_sum = []

    for i in range(len(ratio)):
        if i == 0:
            ratio_sum.append(ratio[i])
        else:
            ratio_sum.append(ratio_sum[-1] + ratio[i])

    print(ratio_sum)
    DT_Fscore_list = []
    RF_Fscore_list = []
    SVM_Fscore_list = []
    LR_Fscore_list = []

    for i in range(len(contribution_list)):

        end = len(ratio_sum) - 1
        for j in range(len(ratio_sum)):
            if ratio_sum[j] > contribution_list[i]:
                end = j
                break
        if end == len(ratio_sum) - 1:
            print("component num is not enough!")

        print("contribution ratio: " + str(contribution_list[i]))
        print("component num: " + str(end + 1))

        DT_Fscore, RF_Fscore, SVM_Fscore, LR_Fscore = model.train_model(X_new[:, 0:end], y_new)

        DT_Fscore_list.append(DT_Fscore)
        RF_Fscore_list.append(RF_Fscore)
        SVM_Fscore_list.append(SVM_Fscore)
        LR_Fscore_list.append(LR_Fscore)

    print("Accuracy of DecisionTree: ")
    print(DT_Fscore_list)
    print("Accuracy of RandomForest: ")
    print(RF_Fscore_list)
    print("Accuracy of SupportVectorMachine: ")
    print(SVM_Fscore_list)
    print("Accuracy of LogisticRegression: ")
    print(LR_Fscore_list)

    # 画图
    l1, = plt.plot(contribution_list, DT_Fscore_list, color='red', linestyle="-", marker=".", linewidth=1)
    l2, = plt.plot(contribution_list, RF_Fscore_list, color='green', linestyle="-", marker=".", linewidth=1)
    l3, = plt.plot(contribution_list, SVM_Fscore_list, color='blue', linestyle="-", marker=".", linewidth=1)
    l4, = plt.plot(contribution_list, LR_Fscore_list, color='yellow', linestyle="-", marker=".", linewidth=1)

    plt.xlabel("contribution_ratio")
    plt.ylabel("Fscore")
    plt.legend(handles=[l1, l2, l3, l4], labels=['DT', 'RF', 'SVM', 'LR'], loc='best')

    plt.savefig(picture_path)
    plt.show()


if __name__ == '__main__':

    maxInt = sys.maxsize
    while True:
        # decrease the maxInt value by factor 10
        # as long as the OverflowError occurs.

        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt / 10)

    # get_feature_score("mic")

    select_feature_and_train_model("mic")

    # PCA("fisher")

    # get_feature_score_overlap()














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
# plt.scatter(X0,Y0,color='red',marker='x')
# plt.scatter(X1,Y1,color='blue',marker='+')
# plt.show()
