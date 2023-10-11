import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import math

np.set_printoptions(threshold=np.inf)


class Net(nn.Module):

    def __init__(self, batch_size):

        super(Net, self).__init__()

        self.batch_size = batch_size

        self.fc1 = nn.Linear(in_features=4, out_features=8)
        self.fc2 = nn.Linear(in_features=8, out_features=2)

    def forward(self, eeg):

        x = eeg
        x = self.fc1(x)
        result = self.fc2(x)

        return result


def Train(train_dataset, test_dataset):

    # 超参数
    batch_size = 4
    batch_print = 4

    # 读取数据集
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

    print("train_samples: %d" % train_dataset.__len__())
    print("test_samples: %d" % test_dataset.__len__())

    # 给网络实例化
    net = Net(batch_size)
    print(net)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 定义更新网络参数的算法（这里用随机梯度下降法）
    optimizer = optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9, last_epoch=-1)

    # 训练
    train_epoch = 1000
    for epoch in range(train_epoch):

        running_loss = 0.0
        finish = False

        for index, data in enumerate(train_loader):

            input_eeg, label = data
            # 调整一下数据类型，不然会报错
            input_eeg = input_eeg.to(torch.float32)
            label = label.to(torch.long)

            # print(input_eeg.shape)
            # print(label.shape)

            optimizer.zero_grad()  # 梯度置0

            output = net(input_eeg)  # 前向传播

            loss = criterion(output, label)  # 计算损失
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新参数
            scheduler.step()

            running_loss += loss.item()
            if index % batch_print == batch_print - 1:  # 每10个batch打印一次结果，并检查损失是否小于阈值
                # print(output)
                print('[%d, %d] loss: %.3f' % (epoch + 1, index + 1, running_loss / batch_print))

                if running_loss / batch_print < 0.05:
                    finish = True
                    break
                running_loss = 0.0

        if finish:
            break

        # 测试
        correct = 0
        total = 0
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
                total += label.size(0)
                correct += (predict == label).sum().item()

        print("test loss: %.3f  test accuracy: %.2f %% " % (test_loss/3, 100 * correct / total))