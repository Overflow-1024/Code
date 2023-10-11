#!/usr/bin/python

import mne
import pymatreader
import numpy as np

def load_all_data(subject='subject 03'):
	
	filename = 'C:/Coding/BCI-Dataset/Berlin/EEG/raw/' + subject + '/with occular artifact/cnt.mat'
	cnt = pymatreader.read_mat(filename)
	filename = 'C:/Coding/BCI-Dataset/Berlin/EEG/raw/' + subject + '/with occular artifact/mrk.mat'
	mrk = pymatreader.read_mat(filename)
	
	all_data, all_label = [], []
	indexs = [0, 2, 4]
	
	for id in indexs:
		info = mne.create_info(cnt['cnt'][id]['clab'][:-2], 200, 'eeg')

		eeg_raw = mne.io.RawArray(cnt['cnt'][id]['x'][...,:-2].T, info)

		mne.set_eeg_reference(eeg_raw)

		eeg_raw.filter(l_freq=8, h_freq=30, picks=['eeg'])

		eeg_events = np.vstack((mrk['mrk'][id]['time']/5.0, np.zeros(mrk['mrk'][id]['time'].shape[0]), mrk['mrk'][id]['event']['desc'])).T.astype('int32')

		eeg_events_id = dict(left=16, right=32)

		eeg_epochs = mne.Epochs(eeg_raw, eeg_events, eeg_events_id, tmin=-5.0, tmax=15.0, baseline=(-3, 0), picks=['eeg'])

#		eeg_epochs = mne.Epochs(eeg_raw, eeg_events, eeg_events_id, tmin=-10.0, tmax=24.995, baseline=(-3, 0), picks=['eeg'])

		all_data.append(eeg_epochs.get_data())

		for label in mrk['mrk'][id]['event']['desc']:
			all_label.append(label//16-1)

	all_data = np.concatenate(all_data, axis=0)
	all_label = np.array(all_label)
	print(all_data.shape)
	print(all_label)
	return all_data, all_label
	
	
def slide_window(data, labels, len, step):
	res_data, res_labels = [], []
	data_len = data[0].shape[1]
	print(data_len)
	for sample, label in zip(data, labels):

		for i in range(0, data_len-len+1, step):
			print(i)
			res_data.append(sample[:,i:i+len])
			res_labels.append(label)

	return np.array(res_data), np.array(res_labels)


# def sliding_window(data, labels, window_size, step):
#
# 	res_data, res_labels = [], []
# 	data_len = data[0].shape[1]
#
# 	for sample, label in zip(data, labels):
#
# 		for i in range(600, data_len-len, step):
# 			res_data.append(sample[:,i:i+len])
# 			res_labels.append(label)
#
# 	return np.array(res_data), np.array(res_labels)