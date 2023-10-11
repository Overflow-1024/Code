import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
import torchinfo
import torchsummary

from Base import CRNN
from Base import EEGNet
from Hybrid import MLB
from Hybrid import LMF
from Attention import Transformer
from Attention import MulTransformer

import Config as cfg

def weights_init(m):

    # 全连接层
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0.0)

    # 卷积层
    if isinstance(m, (nn.Conv2d, nn.Conv3d)):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param, gain=1)
            if 'bias' in name:
                nn.init.constant_(param, 0.0)

    # LSTM层
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param, gain=1)
            if 'bias' in name:
                nn.init.constant_(param, 0.0)

    # 归一化层
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def choose_network(train_dataset, test_dataset, option):


    if option['modal'] == 'eeg':

        if option['net'] == 'EEGNet':
            net = EEGNet.EEGNetClf_eeg(cfg.EEGNet_eeg['net_args'])
            hyper_param = cfg.EEGNet_eeg['hyper_param']

        elif option['net'] == 'CRNN':
            net = CRNN.CRNNClf_eeg(cfg.CRNN_eeg['net_args'])
            hyper_param = cfg.CRNN_eeg['hyper_param']

        else:   # option['net'] == 'Transformer'
            net = Transformer.Transformer(cfg.Transformer_eeg['att_args'], cfg.Transformer_eeg['eeg_args'])
            hyper_param = cfg.Transformer_eeg['hyper_param']

        result = Train_unimodal(net, train_dataset, test_dataset, hyper_param)

    elif option['modal'] == 'nirs':

        if option['net'] == 'EEGNet':
            net = EEGNet.EEGNetClf_nirs(cfg.EEGNet_nirs['net_args'])
            hyper_param = cfg.EEGNet_nirs['hyper_param']

        elif option['net'] == 'CRNN':
            net = CRNN.CRNNClf_nirs(cfg.CRNN_nirs['net_args'])
            hyper_param = cfg.CRNN_nirs['hyper_param']

        else:   # option['net'] == 'Transformer':
            net = Transformer.Transformer(cfg.Transformer_nirs['att_args'], cfg.Transformer_nirs['nirs_args'])
            hyper_param = cfg.Transformer_nirs['hyper_param']

        result = Train_unimodal(net, train_dataset, test_dataset, hyper_param)

    else:   # option['modal'] == 'hybrid'

        if option['fusion'] == 'MLB' and option['net'] == 'EEGNet':
            net = MLB.MLBHybrid_EEGNet(cfg.MLB_EEGNet['eeg_args'],
                                       cfg.MLB_EEGNet['nirs_args'],
                                       cfg.MLB_EEGNet['mlb_args'])
            hyper_param = cfg.MLB_EEGNet['hyper_param']

        elif option['fusion'] == 'MLB' and option['net'] == 'CRNN':
            net = MLB.MLBHybrid_CRNN(cfg.MLB_CRNN['eeg_args'],
                                     cfg.MLB_CRNN['nirs_args'],
                                     cfg.MLB_CRNN['mlb_args'])
            hyper_param = cfg.MLB_CRNN['hyper_param']

        elif option['fusion'] == 'LMF' and option['net'] == 'EEGNet':
            net = LMF.LMFHybrid_EEGNet(cfg.LMF_EEGNet['eeg_args'],
                                       cfg.LMF_EEGNet['nirs_args'],
                                       cfg.LMF_EEGNet['lmf_args'])
            hyper_param = cfg.LMF_EEGNet['hyper_param']

        elif option['fusion'] == 'LMF' and option['net'] == 'CRNN':
            net = LMF.LMFHybrid_CRNN(cfg.LMF_CRNN['eeg_args'],
                                     cfg.LMF_CRNN['nirs_args'],
                                     cfg.LMF_CRNN['lmf_args'])
            hyper_param = cfg.LMF_CRNN['hyper_param']

        else:   # option['fusion'] == 'MulT' and option['net'] == 'Transformer'
            net = MulTransformer.MultimodalTransformer(cfg.MulTransformer['att_args'],
                                                       cfg.MulTransformer['eeg_args'],
                                                       cfg.MulTransformer['nirs_args'], )
            hyper_param = cfg.MulTransformer['hyper_param']

        result = Train_multimodal(net, train_dataset, test_dataset, hyper_param)

    return result


def Train_unimodal(net, train_dataset, test_dataset, param):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    train_batch_size = 24
    batch_print = 10
    test_batch_size = 24

    # 读取数据集
    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=True)

    print("train_samples: %d" % train_dataset.__len__())
    print("test_samples: %d" % test_dataset.__len__())
    print("\n")

    net.to(device)

    # 参数初始化
    net.apply(weights_init)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 定义更新网络参数的算法（这里用Adam算法）
    optimizer = optim.Adam(net.parameters(), lr=param['learning_rate'], weight_decay=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=param['step_size'], gamma=param['gamma'], last_epoch=-1)

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

        # 结束训练的条件
        if train_loss < 0.1 or train_acc > 95:
            end_count += 1
        else:
            end_count = 0
        if end_count >= 10 and epoch >= 100:
            break

        scheduler.step()

    return record


def Train_multimodal(net, train_dataset, test_dataset, param):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    train_batch_size = 24
    batch_print = 10
    test_batch_size = 24

    # 读取数据集
    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=True)

    print("train_samples: %d" % train_dataset.__len__())
    print("test_samples: %d" % test_dataset.__len__())
    print("\n")

    net.to(device)

    # 参数初始化
    net.apply(weights_init)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 定义更新网络参数的算法（这里用Adam算法）
    optimizer = optim.Adam(net.parameters(), lr=param['learning_rate'], weight_decay=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=param['step_size'], gamma=param['gamma'], last_epoch=-1)

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

        # 结束训练的条件
        if train_loss < 0.1 or train_acc > 95:
            end_count += 1
        else:
            end_count = 0
        if end_count >= 10 and epoch >= 100:
            break

        scheduler.step()

    return record
