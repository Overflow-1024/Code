#!/usr/bin/python

from sklearn.utils import shuffle
from others.load_data import load_all_data, slide_window
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.utils.data import DataLoader

class Conv2dWithConstraint(nn.Conv2d):
		def __init__(self, *args, max_norm=1, **kwargs):
				self.max_norm = max_norm
				super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

		def forward(self, x):
				self.weight.data = torch.renorm(
						self.weight.data, p=2, dim=0, maxnorm=self.max_norm
				)
				return super(Conv2dWithConstraint, self).forward(x)


class EEGNet(nn.Module):

		""" Keras Implementation of EEGNet
		http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta

		Note that this implements the newest version of EEGNet and NOT the earlier
		version (version v1 and v2 on arxiv). We strongly recommend using this
		architecture as it performs much better and has nicer properties than
		our earlier version. For example:

				1. Depthwise Convolutions to learn spatial filters within a
				temporal convolution. The use of the depth_multiplier option maps
				exactly to the number of spatial filters learned within a temporal
				filter. This matches the setup of algorithms like FBCSP which learn
				spatial filters within each filter in a filter-bank. This also limits
				the number of free parameters to fit when compared to a fully-connected
				convolution.

				2. Separable Convolutions to learn how to optimally combine spatial
				filters across temporal bands. Separable Convolutions are Depthwise
				Convolutions followed by (1x1) Pointwise Convolutions.


		While the original paper used Dropout, we found that SpatialDropout2D
		sometimes produced slightly better results for classification of ERP
		signals. However, SpatialDropout2D significantly reduced performance
		on the Oscillatory dataset (SMR, BCI-IV Dataset 2A). We recommend using
		the default Dropout in most cases.

		Assumes the input signal is sampled at 128Hz. If you want to use this model
		for any other sampling rate you will need to modify the lengths of temporal
		kernels and average pooling size in blocks 1 and 2 as needed (double the
		kernel lengths for double the sampling rate, etc). Note that we haven't
		tested the model performance with this rule so this may not work well.

		The model with default parameters gives the EEGNet-8,2 model as discussed
		in the paper. This model should do pretty well in general, although it is
	advised to do some model searching to get optimal performance on your
	particular dataset.

		We set F2 = F1 * D (number of input filters = number of output filters) for
		the SeparableConv2D layer. We haven't extensively tested other values of this
		parameter (say, F2 < F1 * D for compressed learning, and F2 > F1 * D for
		overcomplete). We believe the main parameters to focus on are F1 and D.

		Inputs:

			nb_classes      : int, number of classes to classify
			Chans, Samples  : number of channels and time points in the EEG data
			dropoutRate     : dropout fraction
			kernLength      : length of temporal convolution in first layer. We found
												that setting this to be half the sampling rate worked
												well in practice. For the SMR dataset in particular
												since the data was high-passed at 4Hz we used a kernel
												length of 32.
			F1, F2          : number of temporal filters (F1) and number of pointwise
												filters (F2) to learn. Default: F1 = 8, F2 = F1 * D.
			D               : number of spatial filters to learn within each temporal
												convolution. Default: D = 2
			dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.

		"""

		def __init__(self, n_classes, Chans=30, Samples=400, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16, norm_rate=0.25):

				super(EEGNet, self).__init__()
				self.n_classes = n_classes
				self.Chans = Chans
				self.Samples = Samples
				self.dropoutRate = dropoutRate
				self.kernLength = kernLength
				self.F1 = F1
				self.D = D
				self.F2 = F2
				self.norm_rate = norm_rate

				# Size here: (1, C, T)
				self.conv_temporal = nn.Conv2d(in_channels=1, out_channels=self.F1, kernel_size=(1, self.kernLength), stride=(1, 1), padding=(0, self.kernLength//2))
				self.bn_temporal = nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3)
				self.conv_spatial = Conv2dWithConstraint(in_channels=F1, out_channels=self.D*self.F1, kernel_size=(self.Chans, 1), stride=(1, 1), padding=(0, 0))
				# Size here: (D * F1, 1, T)
				self.bn_1 = nn.BatchNorm2d(self.D*self.F1, momentum=0.01, affine=True, eps=1e-3)
				self.pool_1 = nn.AvgPool2d(kernel_size=(1, 5), stride=(1, 5))
				self.drop_1 = nn.Dropout(p=self.dropoutRate)

				# Size here: (D * F1, 1, T // 4)
				self.conv_separable_depth = nn.Conv2d(in_channels=D*F1, out_channels=D*F1, kernel_size=(1, 16), stride=(1, 1), padding=(0, 16//2))
				self.conv_separable_point = nn.Conv2d(in_channels=D*F1, out_channels=F2, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
				# Size here: (F2, 1, T // 4)
				self.bn_2 = nn.BatchNorm2d(F2, momentum=0.01, affine=True, eps=1e-3)
				self.pool_2 = nn.AvgPool2d(kernel_size=(1, 5), stride=(1, 5))
				self.drop_2 = nn.Dropout(p=dropoutRate)

				# Size here: (F2, 1, T // 20)
				self.flat = nn.Flatten()
				self.fc = nn.Linear(in_features=F2*Samples//25, out_features=n_classes)

				_glorot_weight_zero_bias(self)

		def forward(self, x):

				# Size here: (C, T)
				x = x.unsqueeze(1)

				# Size here: (1, C, T)
				x = self.conv_temporal(x)
				x = self.bn_temporal(x)
				x = self.conv_spatial(x)
				# Size here: (1, C, T)
				x = self.bn_1(x)
				x = func.elu(x)
				x = self.pool_1(x)
				x = self.drop_1(x)

				# Size here: (D * F1, 1, T // 4)
				x = self.conv_separable_depth(x)
				x = self.conv_separable_point(x)
				# Size here: (F2, 1, T // 4)
				x = self.bn_2(x)
				x = func.elu(x)
				x = self.pool_2(x)
				x = self.drop_2(x)

				x = self.flat(x)
				out = self.fc(x)

				return out


def _glorot_weight_zero_bias(model):
		"""Initalize parameters of all modules by initializing weights with
		glorot
		 uniform/xavier initialization, and setting biases to zero. Weights from
		 batch norm layers are set to 1.

		Parameters
		----------
		model: Module
		"""
		for module in model.modules():
				if hasattr(module, "weight"):
						if not ("BatchNorm" in module.__class__.__name__):
								nn.init.xavier_uniform_(module.weight, gain=1)
						else:
								nn.init.constant_(module.weight, 1)
				if hasattr(module, "bias"):
						if module.bias is not None:
								nn.init.constant_(module.bias, 0)


def Train(train_dataset, test_dataset):

		# 超参数
		train_batch_size = 32
		batch_print = 2
		test_batch_size = 32

		# 读取数据集
		train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
		test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=True)

		print("train_samples: %d" % train_dataset.__len__())
		print("test_samples: %d" % test_dataset.__len__())
		print("\n")

		# 给网络实例化
		net = EEGNet(n_classes=2)
		print(net)

		# 定义损失函数
		criterion = nn.CrossEntropyLoss()

		# 定义更新网络参数的算法（这里用Adam算法）
		optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-3)
		scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8, last_epoch=-1)

		# 训练
		train_epoch = 300

		for epoch in range(train_epoch):

				train_correct = 0
				train_total = 0
				train_loss = 0.0
				finish = False

				# 训练
				net.train()
				for index, data in enumerate(train_loader):

						input_eeg, label = data
						# 调整一下数据类型，不然会报错
						input_eeg = input_eeg.to(torch.float32)

						label = label.to(torch.long)

						# print(input_eeg.shape)
						# print(input_nirs.shape)
						# print(label.shape)

						optimizer.zero_grad()  # 梯度置0

						output = net(input_eeg)  # 前向传播

						loss = criterion(output, label)  # 计算损失
						loss.backward()  # 反向传播计算梯度
						optimizer.step()  # 更新参数

						train_loss += loss.item()
						_, predict = torch.max(output.data, 1)
						train_total += label.size(0)
						train_correct += (predict == label).sum().item()

						if index % batch_print == batch_print - 1:  # 每10个batch打印一次结果，并检查损失是否小于阈值
								print("[%d, %d] loss: %.3f  train accuracy %.2f%%" % (epoch + 1, index + 1, train_loss / batch_print, 100 * train_correct / train_total))
								if train_loss / batch_print < 0.05:
										finish = True
										break

								train_loss = 0.0

				if finish:
						break

				# 测试
				net.eval()
				test_correct = 0
				test_total = 0
				test_loss = 0.0
				with torch.no_grad():

						for index, data in enumerate(test_loader):
								input_eeg, label = data
								# 调整一下数据类型，不然会报错
								input_eeg = input_eeg.to(torch.float32)

								label = label.to(torch.long)

								output = net(input_eeg)
								loss = criterion(output, label)

								test_loss += loss.item()
								_, predict = torch.max(output.data, 1)
								test_total += label.size(0)
								test_correct += (predict == label).sum().item()

				print("                                                  test loss: %.3f  test accuracy: %.2f%%"
							% (test_loss/(test_dataset.__len__() // test_batch_size), 100 * test_correct / test_total))

				scheduler.step()
				print("learning rate:", optimizer.state_dict()['param_groups'][0]['lr'])



class UniModalDataset(Dataset):  # 需要继承data.Dataset

	def __init__(self, data, label):
		# TODO
		# 1. Initialize file path or list of file names.

		self.data = data
		self.label = label
		self.len = label.shape[0]

		print("Dataset.shape:")
		print("  data.shape: ", data.shape)
		print("  label.shape: ", label.shape)
		print("\n")

		pass

	def __getitem__(self, index):
		# TODO
		# 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
		# 2. Preprocess the data (e.g. torchvision.Transform).
		# 3. Return a data pair (e.g. image and label).

		data = self.data[index, ...]
		label = self.label[index]

		return data, label

		pass

	def __len__(self):
		# You should change 0 to the total size of your dataset.

		return self.len

data, labels = load_all_data('subject 16')
data, labels = shuffle(data, labels, random_state = 42)
data, labels = slide_window(data, labels, 400, 400)
#data, labels = shuffle(data, labels, random_state = 12)

print(data.shape)
print(labels.shape)
division = data.shape[0] * 8 // 10

data_train, data_test = data[:division,...], data[division:,...]
labels_train, labels_test = labels[:division,...], labels[division:,...]
#data_train, labels_train = shuffle(data_train, labels_train, random_state = 42)
#data_test, labels_test = shuffle(data_test, labels_test, random_state = 42)

train_dataset = UniModalDataset(data_train, labels_train)
test_dataset = UniModalDataset(data_test, labels_test)

Train(train_dataset, test_dataset)