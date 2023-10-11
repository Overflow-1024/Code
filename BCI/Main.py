import numpy as np
import pandas as pd
import os
import Util as util
import Processing as proc


# ------------------------  路径  ------------------------

# 笔记本
EEG_raw_rootpath = 'D:/Coding/Data/BCI-Dataset/TU Berlin/EEG/raw'
NIRS_raw_rootpath = 'D:/Coding/Data/BCI-Dataset/TU Berlin/NIRS/NIRS-Hb'

EEG_proc_rootpath = "D:/Coding/Data/BCI-Dataset/TU Berlin/EEG/EEG-proc/proc-bbci"
NIRS_proc_rootpath = "D:/Coding/Data/BCI-Dataset/TU Berlin/NIRS/NIRS-proc"

output_rootpath = 'D:/Coding/Data/BCI-Dataset/result'

# 实验室台式机
# EEG_raw_rootpath = 'C:/Coding/BCI-Dataset/TU Berlin/EEG/raw'
# NIRS_raw_rootpath = 'C:/Coding/BCI-Dataset/TU Berlin/NIRS/NIRS-Hb'
#
# EEG_proc_rootpath = "C:/Coding/BCI-Dataset/TU Berlin/EEG/EEG-proc/proc-bbci"
# NIRS_proc_rootpath = "C:/Coding/BCI-Dataset/TU Berlin/NIRS/NIRS-proc"
#
# output_rootpath = 'C:/Coding/BCI-Dataset/result'

# MatPool服务器
# EEG_raw_rootpath = '../Data/EEG'
# NIRS_raw_rootpath = '../Data/NIRS'
#
# output_rootpath = '../Data/result'


# ------------------------  受试者  ------------------------

# subject_list = [
#     "subject 01", "subject 02", "subject 03", "subject 04", "subject 05", "subject 06", "subject 07", "subject 08", "subject 09", "subject 10",
#     "subject 11", "subject 12", "subject 13", "subject 14", "subject 15", "subject 16", "subject 17", "subject 18", "subject 19", "subject 20",
#     "subject 21", "subject 22", "subject 23", "subject 24", "subject 25", "subject 26", "subject 27", "subject 28", "subject 29"
# ]
subject_list = ["subject 01", "subject 02", "subject 03", "subject 04", "subject 05", "subject 06", "subject 07", "subject 08", "subject 09", "subject 10"]


EEGfs = 200
NIRSfs = 10
'''
原始数据每个trial截取的时间窗：
EEGtime：5000 = 200 × 25[-5, 20]
NIRStime：250 = 10 × 25[-5, 20]
'''

# 生成随机index，用于打乱样本顺序
# shuffle_ix = np.random.permutation(np.arange(60))

def get_EEGdata(net, subject, task):

    # 读取EEG数据并做预处理
    EEGdata, EEGlabel = proc.proc_eeg(EEG_raw_rootpath, subject, task)
    print("\n")
    print("EEG data:")
    print(EEGdata.shape)
    print(EEGlabel.shape)
    print("\n")

    # Size here: trial x time x channel

    # 截取有效时间段
    EEGdata = EEGdata[:, 1100:2900, :]
    print("trial time select:")
    print(EEGdata.shape)
    print("\n")

    # 打乱顺序
    # EEGdata = EEGdata[shuffle_ix]
    # EEGlabel = EEGlabel[shuffle_ix]

    # 滑动时间窗切片
    EEGdata, EEGlabel = util.slide_window(EEGdata, EEGlabel, fs=EEGfs, window_size=2, step=1)
    print("sliding window:")
    print(EEGdata.shape)
    print("\n")

    if net == "CRNN":
        # 电极位置2D表示
        EEGdata = util.reshape_eeg(EEGdata)
        EEGdata = np.transpose(EEGdata, (0, 2, 1, 3, 4))
        print("2D location reshape:")
        print(EEGdata.shape)
        print("\n")

    elif net == 'EEGNet' or net == 'Transformer':
        EEGdata = np.transpose(EEGdata, (0, 2, 1))
        EEGdata = np.expand_dims(EEGdata, axis=1)

    else:
        print("invalid argument")

    # Size for CRNN: sample x time x 1 x channel_height x channel_width
    # Size for EEGNet: sample x 1 x channel x time
    print("input of network")
    print(EEGdata.shape)
    print(EEGlabel.shape)
    print("\n")

    return EEGdata, EEGlabel


def get_NIRSdata(net, subject, task):

    # 读取NIRS数据并做预处理
    NIRSdata, NIRSlabel = proc.proc_nirs(NIRS_raw_rootpath, subject, task)
    print("\n")
    print("NIRS data:")
    print(NIRSdata.shape)
    print(NIRSlabel.shape)
    print("\n")

    # Size here: trial x time x channel

    # 截取有效时间段
    NIRSdata = NIRSdata[:, 55:145, :]
    print("trial time select:")
    print(NIRSdata.shape)
    print("\n")

    # 打乱顺序
    # NIRSdata = NIRSdata[shuffle_ix]
    # NIRSlabel = NIRSlabel[shuffle_ix]

    # 滑动时间窗切片
    NIRSdata, NIRSlabel = util.slide_window(NIRSdata, NIRSlabel, fs=NIRSfs, window_size=2, step=1)
    print("sliding window:")
    print(NIRSdata.shape)
    print("\n")

    if net == "CRNN":
        # 电极位置2D表示
        NIRSdata = util.reshape_nirs(NIRSdata)
        NIRSdata = np.transpose(NIRSdata, (0, 2, 1, 3, 4))
        print("2D location reshape:")
        print(NIRSdata.shape)
        print("\n")

    elif net == 'EEGNet' or net == 'Transformer':
        NIRSdata = np.transpose(NIRSdata, (0, 2, 1))
        data_hbo = NIRSdata[:, :36, :]
        data_hbr = NIRSdata[:, 36:, :]
        NIRSdata = np.stack((data_hbo, data_hbr), axis=1)

    else:
        print("invalid argument")

    # Size for CRNN: sample x time x 2 x channel_height x channel_width
    # Size for EEGNet: sample x 2 x channel x time
    print("input of network")
    print(NIRSdata.shape)
    print(NIRSlabel.shape)
    print("\n")

    return NIRSdata, NIRSlabel


"""
# 读取EEG数据并做预处理

EEGdata, EEGlabel = util.proc_eeg(EEG_raw_rootpath, 'subject 25', task='MA')
print("\n")
print("EEG data:")
print(EEGdata.shape)
print(EEGlabel.shape)
print("\n")
"""

"""
# 读取NIRS数据并做预处理

NIRSdata, NIRSlabel = util.proc_nirs(NIRS_raw_rootpath, 'subject 25', task='MA')
print("\n")
print("NIRS data:")
print(NIRSdata.shape)
print(NIRSlabel.shape)
print("\n")
"""


'''
参数可选值：
task: MI, MA
net: CRNN, EEGNet, Transformer
modal: eeg, nirs, hybrid
fusion: LMF, MLB, MulT
'''


def run(option, output_path):

    if util.check_args(option):

        acc_table = []
        for sub in subject_list:
            if option['modal'] == 'eeg':
                EEGdata, EEGlabel = get_EEGdata(net=option['net'], subject=sub, task=option['task'])
                acc_sub = util.cross_validation_unimodal(EEGdata, EEGlabel, option, sub, output_path)
            elif option['modal'] == 'nirs':
                NIRSdata, NIRSlabel = get_NIRSdata(net=option['net'], subject=sub, task=option['task'])
                acc_sub = util.cross_validation_unimodal(NIRSdata, NIRSlabel, option, sub, output_path)
            else:
                EEGdata, EEGlabel = get_EEGdata(net=option['net'], subject=sub, task=option['task'])
                NIRSdata, NIRSlabel = get_NIRSdata(net=option['net'], subject=sub, task=option['task'])
                acc_sub = util.cross_validation_multimodal(EEGdata, NIRSdata, EEGlabel, option, sub, output_path)
            acc_table.append(acc_sub)

        # 输出统计结果
        df = pd.DataFrame(acc_table, columns=['fold1', 'fold2', 'fold3', 'fold4', 'fold5'], index=subject_list)
        csv_path = os.path.join(output_path, 'acc.csv')
        df.to_csv(csv_path)

    else:
        print("Invalid argument")


# ------------------------  要跑的任务  ------------------------

# Task1  EEGNet/eeg
option = {
    'task': 'MA',
    'modal': 'eeg',
    'net': 'EEGNet',
}
output_path = output_rootpath + '/EEGNet-eeg/MA'
run(option, output_path)

option = {
    'task': 'MI',
    'modal': 'eeg',
    'net': 'EEGNet',
}
output_path = output_rootpath + '/EEGNet-eeg/MI'
run(option, output_path)

# Task2  EEGNet/nirs
option = {
    'task': 'MA',
    'modal': 'nirs',
    'net': 'EEGNet',
}
output_path = output_rootpath + '/EEGNet-nirs/MA'
run(option, output_path)

option = {
    'task': 'MI',
    'modal': 'nirs',
    'net': 'EEGNet',
}
output_path = output_rootpath + '/EEGNet-nirs/MI'
run(option, output_path)

# Task3  CRNN/eeg
option = {
    'task': 'MA',
    'modal': 'eeg',
    'net': 'CRNN',
}
output_path = output_rootpath + '/CRNN-eeg/MA'
run(option, output_path)

option = {
    'task': 'MI',
    'modal': 'eeg',
    'net': 'CRNN',
}
output_path = output_rootpath + '/CRNN-eeg/MI'
run(option, output_path)

# Task4  CRNN/nirs
option = {
    'task': 'MA',
    'modal': 'nirs',
    'net': 'CRNN',
}
output_path = output_rootpath + '/CRNN-nirs/MA'
run(option, output_path)

option = {
    'task': 'MI',
    'modal': 'nirs',
    'net': 'CRNN',
}
output_path = output_rootpath + '/CRNN-nirs/MI'
run(option, output_path)

# Task 5  LMF/EEGNet
option = {
    'task': 'MA',
    'modal': 'hybrid',
    'net': 'EEGNet',
    'fusion': 'LMF'
}
output_path = output_rootpath + '/LMF-EEGNet/MA'
run(option, output_path)

option = {
    'task': 'MI',
    'modal': 'hybrid',
    'net': 'EEGNet',
    'fusion': 'LMF'
}
output_path = output_rootpath + '/LMF-EEGNet/MI'
run(option, output_path)

# Task 6  LMF/CRNN
option = {
    'task': 'MA',
    'modal': 'hybrid',
    'net': 'CRNN',
    'fusion': 'LMF'
}
output_path = output_rootpath + '/LMF-CRNN/MA'
run(option, output_path)

option = {
    'task': 'MI',
    'modal': 'hybrid',
    'net': 'CRNN',
    'fusion': 'LMF'
}
output_path = output_rootpath + '/LMF-CRNN/MI'
run(option, output_path)

# Task 7  MLB/EEGNet
option = {
    'task': 'MA',
    'modal': 'hybrid',
    'net': 'EEGNet',
    'fusion': 'MLB'
}
output_path = output_rootpath + '/MLB-EEGNet/MA'
run(option, output_path)

option = {
    'task': 'MI',
    'modal': 'hybrid',
    'net': 'EEGNet',
    'fusion': 'LMF'
}
output_path = output_rootpath + '/MLB-EEGNet/MI'
run(option, output_path)

# Task 8  MLB/CRNN
option = {
    'task': 'MA',
    'modal': 'hybrid',
    'net': 'CRNN',
    'fusion': 'MLB'
}
output_path = output_rootpath + '/MLB-CRNN/MA'
run(option, output_path)

option = {
    'task': 'MI',
    'modal': 'hybrid',
    'net': 'CRNN',
    'fusion': 'MLB'
}
output_path = output_rootpath + '/MLB-CRNN/MI'
run(option, output_path)

# Task 9  Transformer/eeg
option = {
    'task': 'MA',
    'modal': 'eeg',
    'net': 'Transformer',
}
output_path = output_rootpath + '/Transformer-eeg/MA'
run(option, output_path)

option = {
    'task': 'MI',
    'modal': 'eeg',
    'net': 'Transformer',
}
output_path = output_rootpath + '/Transformer-eeg/MI'
run(option, output_path)

# Task 10  Transformer/nirs
option = {
    'task': 'MA',
    'modal': 'nirs',
    'net': 'Transformer',
}
output_path = output_rootpath + '/Transformer-nirs/MA'
run(option, output_path)

option = {
    'task': 'MI',
    'modal': 'nirs',
    'net': 'Transformer',
}
output_path = output_rootpath + '/Transformer-nirs/MI'
run(option, output_path)

# Task 11  MulTransformer
option = {
    'task': 'MA',
    'modal': 'hybrid',
    'net': 'Transformer',
    'fusion': 'MulT'
}
output_path = output_rootpath + '/MulTransformer/MA'
run(option, output_path)

option = {
    'task': 'MI',
    'modal': 'hybrid',
    'net': 'Transformer',
    'fusion': 'MulT'
}
output_path = output_rootpath + '/MulTransformer/MI'
run(option, output_path)



