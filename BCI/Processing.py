import os
import numpy as np

import mne
import pymatreader
import matplotlib.pyplot as plt

def proc_eeg(rootpath, subject, task):

    cnt_path = os.path.join(rootpath, subject, 'with occular artifact', 'cnt.mat')
    mrk_path = os.path.join(rootpath, subject, 'with occular artifact', 'mrk.mat')

    print(cnt_path)
    print(mrk_path)

    cnt = pymatreader.read_mat(cnt_path)
    mrk = pymatreader.read_mat(mrk_path)

    all_data, all_label = [], []

    if task == 'MI':
        indexs = [0, 2, 4]
    elif task == 'MA':
        indexs = [1, 3, 5]
    else:
        indexs = []

    for id in indexs:

        info = mne.create_info(ch_names=cnt['cnt'][id]['clab'][:-2], ch_types='eeg', sfreq=200)

        eeg_raw = mne.io.RawArray(cnt['cnt'][id]['x'][..., :-2].T, info)

        # # 绘制电极位置
        # montage = mne.channels.read_custom_montage("D:/Coding/Data/BCI-Dataset/TU Berlin/EEG-ChannelLoaction.loc")
        # eeg_raw.set_montage(montage)
        # eeg_raw.plot_sensors(ch_type='eeg', show_names=True)
        # plt.show()
        #
        # # 绘制PSD图
        # eeg_raw.plot_psd()
        # plt.show()

        eeg_raw.filter(l_freq=8, h_freq=30, picks=['eeg'])

        eeg_events = np.vstack((mrk['mrk'][id]['time'] / 5.0, np.zeros(mrk['mrk'][id]['time'].shape[0]),
                                mrk['mrk'][id]['event']['desc'])).T.astype('int32')

        # 绘制events
        # mne.viz.plot_events(eeg_events)

        if task == 'MI':
            eeg_events_id = {'left': 16, 'right': 32}
        else:
            eeg_events_id = {'MA': 16, 'baseline': 32}

        eeg_epochs = mne.Epochs(eeg_raw, eeg_events, eeg_events_id, tmin=-5.0, tmax=20.0, baseline=(-3, 0), picks=['eeg'])

        # 绘制epochs（这个暂时画不了，貌似是因为时间点太多，导致matplotlib发生溢出错误）
        # eeg_epochs.plot(block=True)
        # plt.show()

        all_data.append(eeg_epochs.get_data())

        for label in mrk['mrk'][id]['event']['desc']:
            all_label.append(label // 16 - 1)

    all_data = np.concatenate(all_data, axis=0)
    all_label = np.array(all_label)

    all_data = np.transpose(all_data, (0, 2, 1))

    return all_data, all_label


def proc_nirs(rootpath, subject, task):

    cnt_path = os.path.join(rootpath, subject, 'cntHb.mat')
    mrk_path = os.path.join(rootpath, subject, 'mrk.mat')

    print(cnt_path)
    print(mrk_path)

    cnt = pymatreader.read_mat(cnt_path)
    mrk = pymatreader.read_mat(mrk_path)

    all_data, all_label = [], []

    if task == 'MI':
        indexs = [0, 2, 4]
    elif task == 'MA':
        indexs = [1, 3, 5]
    else:
        indexs = []

    for id in indexs:

        channel_type = ['hbo'] * 36 + ['hbr'] * 36

        info = mne.create_info(ch_names=cnt['cntHb'][id]['clab'], ch_types=channel_type, sfreq=10)

        nirs_raw = mne.io.RawArray(cnt['cntHb'][id]['x'].T, info)

        nirs_raw.filter(l_freq=0.01, h_freq=0.1, picks=['hbo', 'hbr'])

        nirs_events = np.vstack((mrk['mrk'][id]['time'] / 100.0, np.zeros(mrk['mrk'][id]['time'].shape[0]),
                                mrk['mrk'][id]['event']['desc'])).T.astype('int32')

        # 绘制events
        # mne.viz.plot_events(nirs_events)

        if task == 'MI':
            nirs_events_id = {'left': 1, 'right': 2}
        else:
            nirs_events_id = {'MA': 1, 'baseline': 2}

        nirs_epochs = mne.Epochs(nirs_raw, nirs_events, nirs_events_id, tmin=-5.0, tmax=20.0, baseline=(-3, 0), picks=['hbo', 'hbr'])

        # 绘制epochs（这个暂时画不了，貌似是因为时间点太多，导致matplotlib发生溢出错误）
        # nirs_epochs.plot(block=True)
        # plt.show()

        all_data.append(nirs_epochs.get_data())

        for label in mrk['mrk'][id]['event']['desc']:
            all_label.append(label - 1)

    all_data = np.concatenate(all_data, axis=0)
    all_label = np.array(all_label)

    all_data = np.transpose(all_data, (0, 2, 1))

    return all_data, all_label



# -------------------------- EEG------------------------------
# EEG_raw_rootpath = 'D:/Coding/Data/BCI-Dataset/TU Berlin/EEG/raw'
# subject = 'subject 01'
# task = 'MI'
# rootpath = EEG_raw_rootpath
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
# all_data, all_label = [], []
#
# if task == 'MI':
#     indexs = [0, 2, 4]
# elif task == 'MA':
#     indexs = [1, 3, 5]
# else:
#     indexs = []
#
# id = 1
#
# info = mne.create_info(ch_names=cnt['cnt'][id]['clab'][:-2], ch_types='eeg', sfreq=200)
#
# data = cnt['cnt'][id]['x'][..., :-2].T
# data = data / 1000000     # 校准单位，mne库里的电压单位是V
# print(data.shape)
# eeg_raw = mne.io.RawArray(data, info)
#
# # 绘制电极位置
# montage = mne.channels.read_custom_montage("D:/Coding/Data/BCI-Dataset/TU Berlin/EEG-ChannelLoaction.loc")
# eeg_raw.set_montage(montage)
# # eeg_raw.plot_sensors(ch_type='eeg', show_names=True)
# # plt.show()
#
# # # 绘制PSD图
# eeg_raw.plot_psd(average=True)
# # plt.show()
#
# print(eeg_raw.info)
#
# # eeg_raw.plot_psd()
#
# eeg_raw.plot(block=True, duration=5)
#
# eeg_raw.filter(l_freq=8, h_freq=30, picks=['eeg'])
# eeg_raw.plot_psd(average=True)
# eeg_raw.plot(block=True, duration=10)
#
# # fig = eeg_raw.plot(block=True, duration=10)
# # fig.canvas.key_press_event('a')


# -------------------------- NIRS -----------------------

# NIRS_raw_rootpath = 'D:/Coding/Data/BCI-Dataset/TU Berlin/NIRS/NIRS-Hb-my'
# subject = 'subject 10'
# task = 'MA'
# rootpath = NIRS_raw_rootpath
#
# cnt_path = os.path.join(rootpath, subject, 'cntHb-my.mat')
# mrk_path = os.path.join(rootpath, subject, 'mrk.mat')
#
# print(cnt_path)
# print(mrk_path)
#
# cnt = pymatreader.read_mat(cnt_path)
# mrk = pymatreader.read_mat(mrk_path)
#
# all_data, all_label = [], []
#
# if task == 'MI':
#     indexs = [0, 2, 4]
# elif task == 'MA':
#     indexs = [1, 3, 5]
# else:
#     indexs = []
#
# id = 1
#
# # channel_type = ['fnirs_cw_amplitude'] * 72
# # std_ch_names = ['S1_D1 760', 'S2_D1 760', 'S2_D2 760', 'S3_D1 760', 'S3_D2 760', 'S3_D3 760', 'S4_D2 760', 'S4_D3 760',
# #                 'S5_D3 760', 'S6_D4 760', 'S6_D5 760', 'S6_D6 760', 'S7_D7 760', 'S7_D8 760', 'S7_D9 760', 'S8_D8 760',
# #                 'S8_D9 760', 'S8_D10 760', 'S9_D7 760', 'S9_D9 760', 'S9_D11 760', 'S10_D9 760', 'S10_D10 760', 'S10_D11 760',
# #                 'S11_D12 760', 'S11_D13 760', 'S11_D14 760', 'S12_D12 760', 'S12_D14 760', 'S12_D15 760', 'S13_D16 760', 'S13_D13 760',
# #                 'S13_D14 760', 'S14_D16 760', 'S14_D14 760', 'S14_D15 760',
# #                 'S1_D1 850', 'S2_D1 850', 'S2_D2 850', 'S3_D1 850', 'S3_D2 850', 'S3_D3 850', 'S4_D2 850', 'S4_D3 850',
# #                 'S5_D3 850', 'S6_D4 850', 'S6_D5 850', 'S6_D6 850', 'S7_D7 850', 'S7_D8 850', 'S7_D9 850', 'S8_D8 850',
# #                 'S8_D9 850', 'S8_D10 850', 'S9_D7 850', 'S9_D9 850', 'S9_D11 850', 'S10_D9 850', 'S10_D10 850', 'S10_D11 850',
# #                 'S11_D12 850', 'S11_D13 850', 'S11_D14 850', 'S12_D12 850', 'S12_D14 850', 'S12_D15 850', 'S13_D16 850', 'S13_D13 850',
# #                 'S13_D14 850', 'S14_D16 850', 'S14_D14 850', 'S14_D15 850']
#
# channel_type = ['hbo'] * 36 + ['hbr'] * 36
#
# nirs_events = np.vstack((mrk['mrk'][id]['time'] / 100.0, np.zeros(mrk['mrk'][id]['time'].shape[0]),
#                          mrk['mrk'][id]['event']['desc'])).T.astype('int32')
#
# info = mne.create_info(ch_names=cnt['cntHb'][id]['clab'], ch_types=channel_type, sfreq=10)
#
#
# data = cnt['cntHb'][id]['x'].T
# data = data / 1000  # 调整单位
#
# nirs_raw = mne.io.RawArray(data, info)
# print(nirs_raw.info)
#
#
# nirs_raw.plot(block=True, scalings='auto', duration=200)
#
# # raw_od = mne.preprocessing.nirs.optical_density(nirs_raw)
#
# # raw_od.plot(n_channels=len(raw_od.ch_names),
# #             duration=500, show_scrollbars=False)
#
#
# nirs_raw.filter(l_freq=0.01, h_freq=0.2, picks=['hbo', 'hbr'])
#
# nirs_raw.plot(block=True, scalings='auto', duration=200)
#
#
#
# # 绘制events
# mne.viz.plot_events(nirs_events)
#
# if task == 'MI':
#     nirs_events_id = {'left': 1, 'right': 2}
# else:
#     nirs_events_id = {'MA': 1, 'baseline': 2}
#
# reject_criteria = dict(hbo=20e-6)
# nirs_epochs = mne.Epochs(nirs_raw, nirs_events, nirs_events_id, tmin=-5.0, tmax=20.0, baseline=(-3, 0),
#                           picks=['hbo', 'hbr'])
#
#
# nirs_epochs.plot(block=True, scalings='auto')
#
# nirs_epochs['MA'].plot_image(combine='mean', vmin=-10, vmax=10, ts_args=dict(ylim=dict(hbo=[-5, 5], hbr=[-5, 5])))
#
# nirs_epochs['baseline'].plot_image(combine='mean', vmin=-10, vmax=10, ts_args=dict(ylim=dict(hbo=[-5, 5], hbr=[-5, 5])))