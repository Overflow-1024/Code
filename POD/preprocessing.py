import numpy as np
import pandas as pd
import sys
import csv
from imblearn import over_sampling

if __name__ == '__main__':
    data_path = "D:/毕设数据/第一层/ISall.csv"
    valid_data_path = "D:/毕设数据/第一层/ISall_valid.csv"
    NN_data_path = "D:/毕设数据/第一层/ISall_NN.csv"

    in_csv_path = data_path
    out_csv_path = NN_data_path

    maxInt = sys.maxsize
    while True:
        # decrease the maxInt value by factor 10
        # as long as the OverflowError occurs.

        try:
            csv.field_size_limit(maxInt)
            break
        except OverflowError:
            maxInt = int(maxInt / 10)

    # 读取数据文件
    df = pd.read_csv(in_csv_path, engine='python', encoding='gb2312')
    df_key = df.loc[:, ['name', 'pod']]
    df_attr = df.drop(['name', 'pod'], axis=1, inplace=False)
    data_X = df_attr.values
    data_y = df['pod'].values
    attr_name = df_attr.columns.values.tolist()

    # 最多替代的样本数量（两端分别）
    abnormal_num = 3

    # 异常值处理
    for col in attr_name:
        print(col)
        iqr = df_attr[col].quantile(0.75) - df_attr[col].quantile(0.25)
        lower_limit = df_attr[col].quantile(0.25) - 5 * iqr
        upper_limit = df_attr[col].quantile(0.75) + 5 * iqr
        abnormal_min = df_attr[col] < lower_limit
        abnormal_max = df_attr[col] > upper_limit
        print(col + ': ' + str(abnormal_min.sum() + abnormal_max.sum()))

        # 修改极小值
        if abnormal_min.sum() <= abnormal_num:
            df_attr.loc[df_attr[col] < lower_limit, col] = df_attr[col].median()
        else:
            for i in range(abnormal_num):
                df_attr.loc[df_attr[col] == df_attr[col].min(), col] = df_attr[col].median()

        # 修改极大值
        if abnormal_max.sum() <= abnormal_num:
            df_attr.loc[df_attr[col] > upper_limit, col] = df_attr[col].median()
        else:
            for i in range(abnormal_num):
                df_attr.loc[df_attr[col] == df_attr[col].max(), col] = df_attr[col].median()

    # df_valid = pd.merge(df_key, df_attr, how='inner', on=None, left_index=True, right_index=True)
    # df_valid.to_csv(out_csv_path, encoding='gb2312', index=False)

    # print("before drop: ")
    # print(df_attr.shape)
    #
    # df_attr.drop("F0_sma_min_09", axis=1, inplace=True)
    # df_attr.drop("logHNR_sma_percentile1.0_13", axis=1, inplace=True)
    # for attr in attr_name:
    #     if 'Pos' in attr:
    #         df_attr.drop(attr, axis=1, inplace=True)
    #
    # print("after drop: ")
    # print(df_attr.shape)
    #
    # df_attr_scale = (df_attr - df_attr.min()) / (df_attr.max() - df_attr.min())
    #
    # df_scale = pd.merge(df_key, df_attr_scale, how='inner', on=None, left_index=True, right_index=True)
    # df_scale.to_csv(out_csv_path, encoding='gb2312', index=False)


def over_sampling_smote(X, y):

    print("before resample: ")
    print(X.shape)
    print(y.shape)

    smote = over_sampling.SMOTE(sampling_strategy=0.33, k_neighbors=5, kind='regular')
    X_resample, y_resample = smote.fit_resample(X, y)

    print("after resample: ")
    print(X_resample.shape)
    print(y_resample.shape)

    return X_resample, y_resample


def shuffle(X, y):

    permutation = np.random.permutation(y.shape[0])
    X_shuffle = X[permutation, :]
    y_shuffle = y[permutation]

    return X_shuffle, y_shuffle



