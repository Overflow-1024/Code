
import numpy as np
import os
import scipy.io as scio
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn import preprocessing as proc

import MyData
import Network


def load_data(path, source):

    # source=1:论文公开数据集  source=2:CSP提特征

    # 原数据矩阵维度：time × channel × sample
    data_file = scio.loadmat(path)

    data = data_file['data']
    label = data_file['label']

    if source == 1:
        # 调整维度顺序，变为 sample × time × channel
        data = data.transpose(2, 0, 1)
        label = np.squeeze(label)

    if source == 2:
        # 调整维度顺序，变为 sample × time × channel
        data = data.transpose(2, 0, 1)
        label = np.squeeze(label)

        index1 = np.where(label == 1)
        index2 = np.where(label == 2)
        index = np.append(index1, index2)
        index.sort()

        data = data[index, ...]
        label = label[index]

        label = label - 1

    return data, label


def feature_scaling(data, type):
    # 此时data的形式为三维：trial × time × channel
    data_scaling = np.zeros(data.shape)

    for tr_index in range(data.shape[0]):
        data_tr = data[tr_index, :, :]
        if type == 'MinMax':
            data_tr = proc.MinMaxScaler().fit_transform(data_tr)
        elif type == 'Zscore':
            data_tr = proc.scale(data_tr)
        else:
            print("wrong type")
            return -1
        data_scaling[tr_index, :, :] = data_tr

    return data_scaling


def slide_window(data, label, fs, window_size, step):
    # data的形式为 trial × time × channel

    res_data, res_label = [], []
    trial_len = data.shape[1]

    slice_len = fs * window_size
    step_frame = int(step * fs)

    # 验证
    if (trial_len - slice_len) % step_frame > 0:
        print("slide_window error")

    for data_tr, label_tr in zip(data, label):

        for i in range(0, trial_len - slice_len + 1, step_frame):
            res_data.append(data_tr[i:i + slice_len, :])
            res_label.append(label_tr)

    return np.array(res_data), np.array(res_label)


# def sliding_window(data, label, fs, window_size, stride, slice_num):
#     # 在[1,10]s里面切8个2s的segment，[1,3],[2,4],[3,5]...[8,10]
#
#     slice_len = fs * window_size
#     step_frame = int(stride * fs)
#
#     data_slice = np.zeros((data.shape[0] * slice_num, slice_len, data.shape[2]))
#     label_slice = np.zeros((label.shape[0] * slice_num))
#
#     for tr_index in range(data.shape[0]):
#         for sl_index in range(slice_num):
#             data_slice[tr_index * slice_num + sl_index, :, :] = data[tr_index, sl_index * step_frame: sl_index * step_frame + slice_len, :]
#             label_slice[tr_index * slice_num + sl_index] = label[tr_index]
#
#     return data_slice, label_slice


# def moving_average(dataset):
#     window_size = 5
#     slice_len = dataset.dat.shape[1] // window_size
#
#     dat_slice = np.zeros((dataset.dat.shape[0], slice_len, dataset.dat.shape[2], dataset.dat.shape[3]))
#
#     for index in range(dataset.label.shape[0]):
#         for j in range(slice_len):
#             dat = dataset.dat[index, j * window_size: (j + 1) * window_size, :, :]
#             dat_slice[index, j, :, :] = np.mean(dat, axis=0)
#
#     dataset.dat = dat_slice


def reshape_eeg(data):
    # data的形式为 trial × time × channel

    LocationMap = {"AFp1": [1, 7], "AFp2": [1, 9],
                   "AFF5h": [3, 4], "AFF1h": [3, 7], "AFF2h": [3, 9], "AFF6h": [3, 12],
                   "F7": [4, 2], "F3": [4, 5], "F4": [4, 11], "F8": [4, 14],
                   "FCC5h": [7, 3], "FCC3h": [7, 5], "FCC4h": [7, 11], "FCC6h": [7, 13],
                   "T7": [8, 1], "Cz": [8, 8], "T8": [8, 15],
                   "CCP5h": [9, 3], "CCP3h": [9, 5], "CCP4h": [9, 11], "CCP6h": [9, 13],
                   "P7": [12, 2], "P3": [12, 5], "Pz": [12, 8], "P4": [12, 11], "P8": [12, 14],
                   "PPO1h": [13, 7], "PPO2h": [13, 9],
                   "POO1": [15, 7], "POO2": [15, 9]}

    ChannelList = ["F7", "AFF5h", "F3", "AFp1", "AFp2", "AFF6h", "F4", "F8", "AFF1h", "AFF2h",
                   "Cz", "Pz", "FCC5h", "FCC3h", "CCP5h", "CCP3h", "T7", "P7", "P3", "PPO1h",
                   "POO1", "POO2", "PPO2h", "P4", "FCC4h", "FCC6h", "CCP4h", "CCP6h", "P8", "T8"]

    data_out = np.zeros((data.shape[0], data.shape[1], 15, 15), dtype=float)

    for index, channel in enumerate(ChannelList):
        loc = LocationMap[channel]
        data_out[:, :, loc[0]-1, loc[1]-1] = data[:, :, index]

    data_out = np.expand_dims(data_out, axis=2)

    return data_out


def reshape_nirs(data):
    # data的形式为 trial × time × channel

    LocationMap = {"FpzFp1": [1, 7], "FpzFp2": [1, 9],
                   "AF7Fp1": [2, 4], "AF3Fp1": [2, 6], "FpzAFz": [2, 8], "AF4Fp2": [2, 10], "AF8Fp2": [2, 12],
                   "AF3AFz": [3, 7], "AF4AFz": [3, 9],
                   "FC3FC5": [6, 3], "FC3FC1": [6, 5], "FC4FC2": [6, 11], "FC4FC6": [6, 13],
                   "C5FC5": [7, 2], "FC3C3": [7, 4], "C1FC1": [7, 6], "C2FC2": [7, 10], "FC4C4": [7, 12],
                   "C6FC6": [7, 14],
                   "C5C3": [8, 3], "C1C3": [8, 5], "C2C4": [8, 11], "C6C4": [8, 13],
                   "C5CP5": [9, 2], "CP3C3": [9, 4], "C1CP1": [9, 6], "C2CP2": [9, 10], "CP4C4": [9, 12],
                   "C6CP6": [9, 14],
                   "CP3CP5": [10, 3], "CP3CP1": [10, 5], "CP4CP2": [10, 11], "CP4CP6": [10, 13],
                   "OzPOz": [14, 8],
                   "OzO1": [15, 7], "OzO2": [15, 9]}

    # 前36个oxy，后36个deoxy
    ChannelList = ["AF7Fp1", "AF3Fp1", "AF3AFz", "FpzFp1", "FpzAFz", "FpzFp2", "AF4AFz", "AF4Fp2", "AF8Fp2", "OzPOz",
                   "OzO1", "OzO2", "C5CP5", "C5FC5", "C5C3", "FC3FC5", "FC3C3", "FC3FC1", "CP3CP5", "CP3C3",
                   "CP3CP1", "C1C3", "C1FC1", "C1CP1", "C2FC2", "C2CP2", "C2C4", "FC4FC2", "FC4C4", "FC4FC6",
                   "CP4CP6", "CP4CP2", "CP4C4", "C6CP6", "C6C4", "C6FC6"]

    data_out = np.zeros((data.shape[0], data.shape[1], 2, 15, 15), dtype=float)

    for index, channel in enumerate(ChannelList):
        loc = LocationMap[channel]
        data_out[:, :, 0, loc[0]-1, loc[1]-1] = data[:, :, index]
        data_out[:, :, 1, loc[0]-1, loc[1]-1] = data[:, :, index + 36]

    return data_out


# 交叉验证
def cross_validation_multimodal(EEGdata, NIRSdata, label, config, subject, fig_rootpath):

    kf = KFold(n_splits=5, shuffle=False, random_state=None)

    acc_list = []
    count = 0
    for train_index, test_index in kf.split(EEGdata):
        count += 1

        train_data_EEG = EEGdata[train_index, ...]
        test_data_EEG = EEGdata[test_index, ...]

        train_data_NIRS = NIRSdata[train_index, ...]
        test_data_NIRS = NIRSdata[test_index, ...]

        train_label = label[train_index]
        test_label = label[test_index]

        train_dataset = MyData.MultiModalDataset(train_data_EEG, train_data_NIRS, train_label)
        test_dataset = MyData.MultiModalDataset(test_data_EEG, test_data_NIRS, test_label)

        result = Network.choose_network(train_dataset, test_dataset, config)

        fig_name = '{}_{}.png'.format(subject, count)
        fig_path = os.path.join(fig_rootpath, fig_name)
        acc = show_picture(result, fig_path)
        acc_list.append(acc)

    return acc_list


def cross_validation_unimodal(data, label, config, subject, fig_rootpath):

    kf = KFold(n_splits=5, shuffle=False, random_state=None)

    acc_list = []
    count = 0
    for train_index, test_index in kf.split(data):
        count += 1

        train_data = data[train_index, ...]
        test_data = data[test_index, ...]

        train_label = label[train_index]
        test_label = label[test_index]

        train_dataset = MyData.UniModalDataset(train_data, train_label)
        test_dataset = MyData.UniModalDataset(test_data, test_label)

        result = Network.choose_network(train_dataset, test_dataset, config)

        fig_name = '{}_{}.png'.format(subject, count)
        fig_path = os.path.join(fig_rootpath, fig_name)
        acc = show_picture(result, fig_path)
        acc_list.append(acc)

    return acc_list


def show_picture(record, save_path):

    # epoch, acc 是一维数组
    epoch = record['epoch']
    acc_train = record['acc_train']
    acc_test = record['acc_test']
    loss_train = record['loss_train']
    loss_test = record['loss_test']

    # 数据统计
    acc_dict = {}
    for a in acc_test:
        if a in acc_dict:
            acc_dict[a] += 1
        else:
            acc_dict[a] = 1

    acc_list = sorted(acc_dict.items(), key=lambda d: d[0], reverse=True)

    count = 0
    output_acc = 50.0
    threshold = len(epoch) / 8

    for index, item in enumerate(acc_list):
        if index >= 8:
            output_acc = item[0]
            break
        else:
            count += item[1]
            if count >= threshold:
                output_acc = item[0]
                break

    print(output_acc)

    fig = plt.figure(figsize=(16, 16))

    # 画第一张图：epoch-acc
    plt.subplot(2, 2, 1)
    plt.plot(epoch, acc_train, color='blue', label='train')
    plt.plot(epoch, acc_test, color='red', label='test')
    plt.axhline(y=output_acc, color='green', linestyle='dotted')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.legend()

    # 画第二张图：epoch-loss
    plt.subplot(2, 2, 2)
    plt.plot(epoch, loss_train, color='blue', label='train')
    plt.plot(epoch, loss_test, color='red', label='test')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    # 画第三张图：acc-acc出现次数
    if len(acc_list) > 8:
        acc_list = acc_list[:8]

    temp = list(map(list, zip(*acc_list)))
    x = temp[0]
    height = temp[1]

    plt.subplot(2, 2, 3)
    plt.bar(x, height)
    plt.xlabel('acc')
    plt.ylabel('count')

    # plt.show()
    plt.savefig(save_path)

    return output_acc


# 检查参数合法性
def check_args(config):
    task_set = {'MI', 'MA'}
    modal_set = {'eeg', 'nirs', 'hybrid'}
    net_set = {'EEGNet', 'CRNN', 'Transformer'}
    fusion_set = {'LMF', 'MLB', 'MulT'}

    if 'task' in config:
        if config['task'] not in task_set:
            return False
    if 'modal' in config:
        if config['modal'] not in modal_set:
            return False
    if 'net' in config:
        if config['net'] not in net_set:
            return False
    if 'fusion' in config:
        if config['fusion'] not in fusion_set:
            return False

    return True