import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary

import Util as util
import Config as cfg

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)









class EEGNet_eeg(nn.Module):

    """

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

    def __init__(self, n_classes=2, Chans=64, Samples=128, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16, norm_rate=0.25):

        super(EEGNet_eeg, self).__init__()
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
        self.conv_temporal = nn.Conv2d(in_channels=1, out_channels=self.F1, kernel_size=(1, self.kernLength), stride=(1, 1), padding=(0, (self.kernLength-1)//2), bias=False)
        self.bn_temporal = nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3)
        self.conv_spatial = Conv2dWithConstraint(in_channels=self.F1, out_channels=self.D*self.F1, kernel_size=(self.Chans, 1), stride=(1, 1), padding=(0, 0), groups=self.F1, bias=False, max_norm=1)

        # Size here: (D * F1, 1, T)
        self.bn_1 = nn.BatchNorm2d(self.D*self.F1, momentum=0.01, affine=True, eps=1e-3)
        self.pool_1 = nn.AvgPool2d(kernel_size=(1, 5), stride=(1, 5))   # 200/5=40Hz
        self.drop_1 = nn.Dropout(p=self.dropoutRate)

        # Size here: (D * F1, 1, T // 5)
        self.conv_separable_depth = nn.Conv2d(in_channels=self.D*self.F1, out_channels=self.D*self.F1, kernel_size=(1, 21), stride=(1, 1), padding=(0, (21-1)//2), groups=self.D*self.F1, bias=False)
        self.conv_separable_point = nn.Conv2d(in_channels=self.D*self.F1, out_channels=self.F2, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        # Size here: (F2, 1, T // 5)
        self.bn_2 = nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3)
        self.pool_2 = nn.AvgPool2d(kernel_size=(1, 20), stride=(1, 20))   #
        self.drop_2 = nn.Dropout(p=self.dropoutRate)

        # Size here: (F2, 1, T // 100)
        self.flat = nn.Flatten()

    def forward(self, x):

        # Size here: (1, C, T)
        x = self.conv_temporal(x)
        x = self.bn_temporal(x)
        x = self.conv_spatial(x)
        # Size here: (D * F1, 1, T)
        x = self.bn_1(x)
        x = func.elu(x)
        x = self.pool_1(x)
        x = self.drop_1(x)

        # Size here: (D * F1, 1, T // 5)
        x = self.conv_separable_depth(x)
        x = self.conv_separable_point(x)
        # Size here: (F2, 1, T // 5)
        x = self.bn_2(x)
        x = func.elu(x)
        x = self.pool_2(x)
        x = self.drop_2(x)

        # Size here: (F2, 1, T // 100)
        out = self.flat(x)
        # out = self.fc(x)
        # print(out.shape)

        return out


class EEGNetClf_eeg(nn.Module):

    def __init__(self, args):

        super(EEGNetClf_eeg, self).__init__()

        self.feature_eeg = EEGNet_eeg(n_classes=2, Chans=args['channel'], Samples=args['time'], kernLength=args['kernlength'])
        self.fc = nn.Linear(in_features=args['out_dim'], out_features=2)

    def forward(self, x):

        x = self.feature_eeg(x)
        out = self.fc(x)

        return out








class EEGNet_nirs(nn.Module):

    """

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

    def __init__(self, n_classes=2, Chans=64, Samples=128, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16, norm_rate=0.25):

        super(EEGNet_nirs, self).__init__()
        self.n_classes = n_classes
        self.Chans = Chans
        self.Samples = Samples
        self.dropoutRate = dropoutRate
        self.kernLength = kernLength
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.norm_rate = norm_rate

        # Size here: (2, C, T)
        self.conv_temporal = nn.Conv2d(in_channels=2, out_channels=self.F1, kernel_size=(1, self.kernLength), stride=(1, 1), padding=(0, (self.kernLength-1)//2), bias=False)
        self.bn_temporal = nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3)
        self.conv_spatial = Conv2dWithConstraint(in_channels=self.F1, out_channels=self.D*self.F1, kernel_size=(self.Chans, 1), stride=(1, 1), padding=(0, 0), groups=self.F1, bias=False, max_norm=1)

        # Size here: (D * F1, 1, T)
        self.bn_1 = nn.BatchNorm2d(self.D*self.F1, momentum=0.01, affine=True, eps=1e-3)
        self.pool_1 = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.drop_1 = nn.Dropout(p=self.dropoutRate)

        # Size here: (D * F1, 1, T // 2)
        self.conv_separable_depth = nn.Conv2d(in_channels=self.D*self.F1, out_channels=self.D*self.F1, kernel_size=(1, 5), stride=(1, 1), padding=(0, (5-1)//2), groups=self.D*self.F1, bias=False)
        self.conv_separable_point = nn.Conv2d(in_channels=self.D*self.F1, out_channels=self.F2, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        # Size here: (F2, 1, T // 2)
        self.bn_2 = nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3)
        self.pool_2 = nn.AvgPool2d(kernel_size=(1, 5), stride=(1, 5))
        self.drop_2 = nn.Dropout(p=self.dropoutRate)

        # Size here: (F2, 1, T // 10)
        self.flat = nn.Flatten()
        # self.fc = nn.Linear(in_features=self.F2*self.Samples//5, out_features=self.n_classes)

        # _weights_init(self)

    def forward(self, x):

        # Size here: (2, C, T)
        x = self.conv_temporal(x)
        x = self.bn_temporal(x)
        x = self.conv_spatial(x)
        # Size here: (D * F1, 1, T)
        x = self.bn_1(x)
        x = func.elu(x)
        x = self.pool_1(x)
        x = self.drop_1(x)

        # Size here: (D * F1, 1, T // 2)
        x = self.conv_separable_depth(x)
        x = self.conv_separable_point(x)
        # Size here: (F2, 1, T // 2)
        x = self.bn_2(x)
        x = func.elu(x)
        x = self.pool_2(x)
        x = self.drop_2(x)

        # Size here: (F2, 1, T // 10)
        out = self.flat(x)
        # out = self.fc(x)
        # print(out.shape)

        return out


class EEGNetClf_nirs(nn.Module):

    def __init__(self, args):

        super(EEGNetClf_nirs, self).__init__()

        self.feature_nirs = EEGNet_nirs(n_classes=2, Chans=args['channel'], Samples=args['time'], kernLength=args['kernlength'])
        self.fc = nn.Linear(in_features=args['out_dim'], out_features=2)

    def forward(self, x):

        x = self.feature_nirs(x)
        out = self.fc(x)

        return out






def weights_init(model):
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


def Train(train_dataset, test_dataset, modal):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 读取参数
    config_eeg = cfg.EEGNet_eeg
    config_nirs = cfg.EEGNet_nirs

    train_batch_size = 12
    batch_print = 10
    test_batch_size = 12

    # 读取数据集
    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=True)

    print("train_samples: %d" % train_dataset.__len__())
    print("test_samples: %d" % test_dataset.__len__())
    print("\n")

    # 给网络实例化
    if modal == 'eeg':
        args = config_eeg['net_args']
        hyper_param = config_eeg['hyper_param']

        net = EEGNetClf_eeg(args)
        print(net)
        summary(net, input_size=(1, args['channel'], args['time']))
    else:
        args = config_nirs['net_args']
        hyper_param = config_nirs['hyper_param']

        net = EEGNetClf_nirs(args)
        print(net)
        summary(net, input_size=(2, args['channel'], args['time']))

    net.to(device)

    # 参数初始化
    net.apply(weights_init)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 定义更新网络参数的算法（这里用Adam算法）
    optimizer = optim.Adam(net.parameters(), lr=hyper_param['learning_rate'], weight_decay=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=hyper_param['step_size'], gamma=hyper_param['gamma'], last_epoch=-1)

    # 训练
    train_epoch = 200
    end_count = 0

    # 画图用
    record = {
        'epoch': [],
        'acc_train': [],
        'acc_test': [],
        'loss_train': [],
        'loss_test': []
    }

    for epoch in range(train_epoch):

        train_loss = 0.0

        print("learning rate:", optimizer.state_dict()['param_groups'][0]['lr'])

        # 训练
        net.train()

        for index, data in enumerate(train_loader):

            input, label = data

            # 调整一下数据类型，不然会报错
            input = input.to(torch.float32)
            label = label.to(torch.long)

            input = input.to(device)
            label = label.to(device)

            # print(input.shape)
            # print(label.shape)

            optimizer.zero_grad()  # 梯度置0

            output = net(input)  # 前向传播

            loss = criterion(output, label)  # 计算损失

            loss.backward()  # 反向传播计算梯度

            optimizer.step()  # 更新参数

            train_loss += loss.item()

            _, predict = torch.max(output.data, 1)

            if index % batch_print == batch_print - 1:  # 每n个batch打印一次结果，并检查损失是否小于阈值
                print("[%d, %d] loss: %.3f" % (epoch + 1, index + 1, train_loss / batch_print))
                train_loss = 0.0

        # 测试
        net.eval()

        with torch.no_grad():

            def get_acc(loader):

                correct = 0
                total = 0
                total_loss = 0.0

                for index, data in enumerate(loader):

                    input, label = data

                    # 调整一下数据类型，不然会报错
                    input = input.to(torch.float32)
                    label = label.to(torch.long)

                    input = input.to(device)
                    label = label.to(device)

                    output = net(input)

                    loss = criterion(output, label)

                    total_loss += loss.item()
                    
                    _, predict = torch.max(output.data, 1)
                    total += label.size(0)
                    correct += (predict == label).sum().item()

                return total_loss, 100 * correct / total

            train_loss, train_acc = get_acc(train_loader)
            test_loss, test_acc = get_acc(test_loader)

            # 前面的loss是合计的，所有batch加起来的loss，要做除法还原一下
            train_loss = train_loss/(train_dataset.__len__() // train_batch_size)
            test_loss = test_loss/(test_dataset.__len__() // test_batch_size)

        # 保存画图数据
        record['epoch'].append(epoch + 1)
        record['acc_train'].append(train_acc)
        record['acc_test'].append(test_acc)
        record['loss_train'].append(train_loss)
        record['loss_test'].append(test_loss)

        print("train loss: %.3f  train accuracy: %.2f%%  test loss: %.3f  test accuracy: %.2f%% \n"
              % (train_loss, train_acc, test_loss, test_acc))

        # 如果连续10个epoch loss<0.1，则结束训练
        if train_loss < 0.1 or train_acc > 95:
            end_count += 1
        else:
            end_count = 0
        if end_count >= 10:
            break

        scheduler.step()

    return record



