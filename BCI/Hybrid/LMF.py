import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary

from Base import EEGNet
from Base import CRNN

import Util as util
import Config as cfg


# Low-rank Multimodal Fusion
class LMF(nn.Module):

    def __init__(self, eeg_dim, nirs_dim, out_dim, rank):

        super(LMF, self).__init__()

        self.eegDim = eeg_dim
        self.nirsDim = nirs_dim
        self.outDim = out_dim
        self.rank = rank

        self.eeg_factor = nn.Parameter(torch.Tensor(self.rank, self.eegDim + 1, self.outDim))
        self.nirs_factor = nn.Parameter(torch.Tensor(self.rank, self.nirsDim + 1, self.outDim))
        self.fusion_weights = nn.Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = nn.Parameter(torch.Tensor(1, self.outDim))

        # 初始化
        nn.init.xavier_uniform_(self.eeg_factor, gain=1)
        nn.init.xavier_uniform_(self.nirs_factor, gain=1)
        nn.init.xavier_uniform_(self.fusion_weights, gain=1)
        nn.init.constant_(self.fusion_bias, 0.0)

    def forward(self, eeg_input, nirs_input):

        # 在input向量后面concat一个1
        batch_size = eeg_input.shape[0]
        if cfg.Mode == 'gpu':
            one_tensor = torch.ones(batch_size, 1).cuda()
        else:
            one_tensor = torch.ones(batch_size, 1)

        e = torch.cat((one_tensor, eeg_input), dim=1)
        n = torch.cat((one_tensor, nirs_input), dim=1)

        # here is a more efficient implementation than the one the paper describes
        # basically swapping the order of summation and elementwise product

        fusion_e = torch.matmul(e, self.eeg_factor)
        fusion_n = torch.matmul(n, self.nirs_factor)
        fusion_zy = fusion_e * fusion_n
        # 这里乘完之后 矩阵fusion_zy 的维度: rank × batch × outdim

        # use linear transformation instead of simple summation, more flexibility
        # output = torch.sum(fusion_zy, dim=0).squeeze()
        output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.outDim)

        return output


class LMFHybrid_EEGNet(nn.Module):

    def __init__(self, args_eeg, args_nirs, args_lmf):

        super(LMFHybrid_EEGNet, self).__init__()

        self.feature_eeg = EEGNet.EEGNet_eeg(n_classes=2, Chans=args_eeg['channel'], Samples=args_eeg['time'], kernLength=args_eeg['kernlength'])

        self.feature_nirs = EEGNet.EEGNet_nirs(n_classes=2, Chans=args_nirs['channel'], Samples=args_nirs['time'], kernLength=args_nirs['kernlength'])

        self.hybrid = LMF(eeg_dim=args_lmf['eeg_dim'], nirs_dim=args_lmf['nirs_dim'], out_dim=args_lmf['out_dim'], rank=args_lmf['rank'])

        self.fc = nn.Linear(in_features=args_lmf['out_dim'], out_features=2)

    def forward(self, eeg, nirs):

        e = self.feature_eeg(eeg)
        n = self.feature_nirs(nirs)

        out = self.hybrid(e, n)
        out = self.fc(out)

        return out


class LMFHybrid_CRNN(nn.Module):

    def __init__(self, args_eeg, args_nirs, args_lmf):

        super(LMFHybrid_CRNN, self).__init__()

        self.feature_eeg = CRNN.CRNN_eeg()

        self.feature_nirs = CRNN.CRNN_nirs()

        self.hybrid = LMF(eeg_dim=args_lmf['eeg_dim'], nirs_dim=args_lmf['nirs_dim'], out_dim=args_lmf['out_dim'], rank=args_lmf['rank'])

        self.fc = nn.Linear(in_features=args_lmf['out_dim'], out_features=2)

    def forward(self, eeg, nirs):

        e = self.feature_eeg(eeg)
        n = self.feature_nirs(nirs)

        out = self.hybrid(e, n)
        out = self.fc(out)

        return out


def Train(train_dataset, test_dataset, base):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 读取参数
    config_EEGNet = cfg.LMF_EEGNet
    config_CRNN = cfg.LMF_CRNN

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
    if base == 'EEGNet':
        eeg_args = config_EEGNet['eeg_args']
        nirs_args = config_EEGNet['nirs_args']
        lmf_args = config_EEGNet['lmf_args']
        hyper_param = config_EEGNet['hyper_param']

        net = LMFHybrid_EEGNet(eeg_args, nirs_args, lmf_args)
        print(net)

    else:
        eeg_args = config_CRNN['eeg_args']
        nirs_args = config_CRNN['nirs_args']
        lmf_args = config_CRNN['lmf_args']
        hyper_param = config_CRNN['hyper_param']

        net = LMFHybrid_CRNN(eeg_args, nirs_args, lmf_args)
        print(net)

    net.to(device)

    # 参数初始化
    if base == 'EEGNet':
        net.apply(EEGNet.weights_init)
    else:
        net.apply(CRNN.weights_init)

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

            input_eeg, input_nirs, label = data

            # 调整一下数据类型，不然会报错
            input_eeg = input_eeg.to(torch.float32)
            input_nirs = input_nirs.to(torch.float32)
            label = label.to(torch.long)

            input_eeg = input_eeg.to(device)
            input_nirs = input_nirs.to(device)
            label = label.to(device)

            # print(input_eeg.shape)
            # print(input_nirs.shape)
            # print(label.shape)

            optimizer.zero_grad()  # 梯度置0

            output = net(input_eeg, input_nirs)  # 前向传播

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

                    input_eeg, input_nirs, label = data

                    # 调整一下数据类型，不然会报错
                    input_eeg = input_eeg.to(torch.float32)
                    input_nirs = input_nirs.to(torch.float32)
                    label = label.to(torch.long)

                    input_eeg = input_eeg.to(device)
                    input_nirs = input_nirs.to(device)
                    label = label.to(device)

                    output = net(input_eeg, input_nirs)

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
