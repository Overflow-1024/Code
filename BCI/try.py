import mne
import pymatreader
import os
import matplotlib.pyplot as plt
import numpy as np

# rootpath = 'C:/Coding/BCI-Dataset/TU Berlin/EEG/raw'
# subject = 'subject 25'
# task = 'MA'
# id = 1
#
# cnt_path = os.path.join(rootpath, subject, 'with occular artifact', 'cnt.mat')
# mrk_path = os.path.join(rootpath, subject, 'with occular artifact', 'mrk.mat')
#
# print(cnt_path)
# print(mrk_path)
#
# cnt = pymatreader.read_mat(cnt_path)
# mrk = pymatreader.read_mat(mrk_path)
#
# info = mne.create_info(ch_names=cnt['cnt'][id]['clab'][:-2], ch_types='eeg', sfreq=200)
#
# eeg_raw = mne.io.RawArray(cnt['cnt'][id]['x'][..., :-2].T, info)
#
# # 绘制电极位置
# montage = mne.channels.read_custom_montage("C:/Coding/BCI-Dataset/TU Berlin/EEG-ChannelLoaction.loc")
# eeg_raw.set_montage(montage)
# eeg_raw.plot_sensors()
# plt.show()
#
# # 绘制PSD图
# eeg_raw.plot_psd()
# plt.show()
#
# eeg_raw.filter(l_freq=8, h_freq=30, picks=['eeg'])
#
# eeg_events = np.vstack((mrk['mrk'][id]['time'] / 5.0, np.zeros(mrk['mrk'][id]['time'].shape[0]),
#                         mrk['mrk'][id]['event']['desc'])).T.astype('int32')
# # 绘制events
# mne.viz.plot_events(eeg_events)
#
# if task == 'MI':
#     eeg_events_id = {'left': 16, 'right': 32}
# else:
#     eeg_events_id = {'MA': 16, 'baseline': 32}
#
# eeg_epochs = mne.Epochs(eeg_raw, eeg_events, eeg_events_id, tmin=-5.0, tmax=15.0, baseline=(-3, 0), picks=['eeg'])
#
# # # 绘制epochs
# # eeg_epochs.plot(block=True)
# # plt.show()
#
#
# all_label = []
# all_data = eeg_epochs.get_data()
#
# for label in mrk['mrk'][id]['event']['desc']:
#     all_label.append(label // 16 - 1)
#
# all_data = np.array(all_data)
# all_label = np.array(all_label)
#
# all_data = np.transpose(all_data, (0, 2, 1))
#
# print(all_data.shape)
# print(all_label.shape)

import torch
from Attention import SEblock
from Attention import Transformer

# batch = 20
# channel = 16
# x = torch.ones(batch, channel, 40, 40)
# print(x.shape)
# se = SEblock.SEblock(channels=channel, ratio=4)
#
# y = se(x)
#
# print(y.shape)

# batch = 20
# time = 20
# emb = 16
# h = 4
#
# x = torch.ones(batch, time, emb)
# print(x.shape)
# transf = Transformer.MultiHeadAttention(emb_dim=16, head=h)
# y = transf(x)
#
# print(y.shape)

# a = torch.ones(1)
# print(a)
# b = a.repeat(4, 1)
# print(b.shape)
# print(b)
#
# state = np.random.get_state()

# np.random.shuffle(a)
# np.random.set_state(state)
# np.random.shuffle(b)
#
# print(a)
# print(b)
#
# np.random.set_state(state)
# np.random.shuffle(c)
# print(c)
#
# np.random.set_state(state)
# np.random.shuffle(d)
# print(d)

# a = np.array([1, 2, 3, 4, 5])
# b = np.array([11, 12, 13, 14, 15])
# c = np.array([21, 22, 23, 24, 25])
# d = np.array([31, 32, 33, 34, 35])
#
# shuffle_ix = np.random.permutation(np.arange(5))
# print(shuffle_ix)
# print(type(shuffle_ix))
# shuffle_a = a[shuffle_ix]
# shuffle_b = b[shuffle_ix]
# shuffle_c = c[shuffle_ix]
#
# print(shuffle_a)
# print(shuffle_b)
# print(shuffle_c)


# wl = 760
# age = 28.5
# a = 223.3
# b = 0.05624
# age_power = 0.8493
# c = -5.723e-7
# d = 0.001245
# e = -0.9025
#
# DPF = a + b * age ** age_power + c * wl ** 3 + d * wl ** 2 + e * wl
# print(DPF)

input1_size = 10
input2_size = 8
output_size = 5
rank = 3

weight1 = torch.rand(rank, input1_size, output_size)
weight2 = torch.rand(rank, input1_size, output_size)

input1 = torch.randn(10)  # 输入特征向量 1 的大小为 16x10  16为batch_size
input2 = torch.randn(8)  # 输入特征向量 2 的大小为 16x8




print(input1.shape)
print(weight1.shape)

# print(input2.shape)
# print(weight2.T.shape)

c1 = torch.matmul(input1, weight1)
# c1 = torch.matmul(weight1.T, input1)


print(c1.shape)





# a = torch.ones(10, 3, 4)
# b = torch.ones(4, 3)
#
# c = torch.matmul(a, b)
# print(c.shape)
# print(c)
