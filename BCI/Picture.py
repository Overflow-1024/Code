import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 画折线图
def picture_plot():
    root_path = "D:/Coding/Data/BCI-Dataset/result/result-最新"
    file_name = "结果汇总-MA任务-use-融合统计.xlsx"

    out_root_path = "D:/学习/毕业论文/小论文"
    picture_name = "特征融合-MA任务-折线图.png"

    file_path = os.path.join(root_path, file_name)
    df = pd.read_excel(file_path)

    df_data = df.values
    df_data = df_data[1:21, 1:]

    sub = np.arange(1, 21)
    CRNN_eeg = df_data[:, 0]
    CRNN_fnirs = df_data[:, 1]
    CNN_eeg = df_data[:, 2]
    CNN_fnirs = df_data[:, 3]

    EEG = df_data[:, 4]
    fNIRS = df_data[:, 5]
    LMF_CNN = df_data[:, 7]
    LMF = df_data[:, 8]
    MLB = df_data[:, 11]

    EEG_Att = df_data[:, 12]
    fNIRS_Att = df_data[:, 13]
    MulT = df_data[:, 14]

    fig = plt.figure()

    # plt.plot(sub, CNN_eeg, color='mediumseagreen', label='CNN_eeg', marker='.')
    # plt.plot(sub, CNN_fnirs, color='royalblue', label='CNN_fnirs', marker='*')
    # plt.plot(sub, CRNN_eeg, color='red', label='CRNN_eeg', marker='+')
    # plt.plot(sub, CRNN_fnirs, color='gold', label='CRNN_fnirs', marker='x')

    plt.plot(sub, CNN_eeg, color='royalblue', label='EEG', marker='.')
    plt.plot(sub, CNN_fnirs, color='red', label='fNIRS', marker='*')
    plt.plot(sub, LMF_CNN, color='mediumseagreen', label='LMF', marker='+')
    # plt.plot(sub, MLB, color='royalblue', label='MLB', marker='x')

    # plt.plot(sub, EEG_Att, color='mediumseagreen', label='EEG_Att', marker='+')
    # plt.plot(sub, fNIRS_Att, color='red', label='fNIRS_Att', marker='*')
    # plt.plot(sub, MulT, color='royalblue', label='MulT', marker='x')

    x_ticks = np.arange(0, 21, 5)[1:]
    y_ticks = np.arange(0, 110, 10)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)

    plt.xlabel('Subject')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    out_path = os.path.join(out_root_path, picture_name)
    plt.savefig(out_path)
    plt.show()


# 画柱状图
def picture_bar():

    def get_data(root_path, file_name):

        file_path = os.path.join(root_path, file_name)
        df = pd.read_excel(file_path)

        df_data = df.values
        df_data = df_data[1:, 1:]

        CNN_eeg = df_data[21, 2]
        CNN_fnirs = df_data[21, 3]
        LMF_CNN = df_data[21, 7]

        EEG = df_data[21, 4]
        fNIRS = df_data[21, 5]
        LMF = df_data[21, 8]
        MLB = df_data[21, 11]

        EEG_Att = df_data[21, 12]
        fNIRS_Att = df_data[21, 13]
        MulT = df_data[21, 14]

        return EEG, fNIRS, LMF, MLB, EEG_Att, fNIRS_Att, MulT

    root_path = "D:/Coding/Data/BCI-Dataset/result/result-最新"
    file_name_MA = "结果汇总-MA任务-use-融合统计.xlsx"
    file_name_MI = "结果汇总-MI任务-use-融合统计.xlsx"

    out_root_path = "D:/学习/毕业论文/小论文"
    picture_name = "融合与不融合对比-柱状图.png"

    data_MA = get_data(root_path, file_name_MA)
    data_MI = get_data(root_path, file_name_MI)

    EEG = [data_MA[0], data_MI[0]]
    fNIRS = [data_MA[1], data_MI[1]]
    LMF = [data_MA[2], data_MI[2]]
    MLB = [data_MA[3], data_MI[3]]

    EEG_Att = [data_MA[4], data_MI[4]]
    fNIRS_Att = [data_MA[5], data_MI[5]]
    MulT = [data_MA[6], data_MI[6]]

    fig = plt.figure()

    bar_width = 0.5
    space = 0.1

    x = np.array([3, 6])
    x1 = x - 1.5 * (bar_width + space)
    x2 = x - 0.5 * (bar_width + space)
    x3 = x + 0.5 * (bar_width + space)
    x4 = x + 1.5 * (bar_width + space)

    plt.bar(x1, EEG, width=bar_width, label='EEG', color='royalblue', zorder=2)
    plt.bar(x2, fNIRS, width=bar_width, label='fNIRS', color='red', zorder=2)
    plt.bar(x3, LMF, width=bar_width, label='LMF', color='mediumseagreen', zorder=2)


    # plt.bar(x1, EEG, width=bar_width, label='EEG', color='gold', zorder=2)
    # plt.bar(x2, fNIRS, width=bar_width, label='fNIRS', color='sienna', zorder=2)
    # plt.bar(x3, LMF, width=bar_width, label='LMF', color='mediumseagreen', zorder=2)
    # plt.bar(x4, MLB, width=bar_width, label='MLB', color='royalblue', zorder=2)

    # x = np.array([3, 6])
    # x1 = x - (bar_width + space)
    # x2 = x
    # x3 = x + (bar_width + space)
    #
    # plt.bar(x1, EEG_Att, width=bar_width, label='EEG_Att', color='gold', zorder=2)
    # plt.bar(x2, fNIRS_Att, width=bar_width, label='fNIRS_Att', color='mediumseagreen', zorder=2)
    # plt.bar(x3, MulT, width=bar_width, label='MulT', color='royalblue', zorder=2)

    # x = np.array([3, 6])
    # x1 = x - (bar_width + space)
    # x2 = x
    # x3 = x + (bar_width + space)
    #
    # plt.bar(x1, LMF, width=bar_width, label='LMF', color='mediumseagreen', zorder=2)
    # plt.bar(x2, MLB, width=bar_width, label='MLB', color='red', zorder=2)
    # plt.bar(x3, MulT, width=bar_width, label='MulT', color='royalblue', zorder=2)

    x_ticks = ['MA', 'MI']
    y_ticks = np.arange(0, 110, 10)

    plt.xticks(x, x_ticks)
    plt.yticks(y_ticks)
    plt.ylabel('Accuracy (%)')

    plt.xlim(1, 9)

    plt.legend(loc='best')
    plt.grid(linestyle='--', axis='y')

    out_path = os.path.join(out_root_path, picture_name)
    plt.savefig(out_path)
    plt.show()


if __name__ == '__main__':
    picture_bar()