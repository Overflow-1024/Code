import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary

from sklearn import svm
from sklearn.metrics import accuracy_score

from Base import EEGNet
from Base import CRNN
import Util as util


def svm_classify(train_data, train_label, test_data, test_label):
    """
    trains a linear SVM on the data
    input C specifies the penalty factor of SVM
    """

    # print('training SVM...')
    clf = svm.LinearSVC(C=0.01, dual=False)
    clf.fit(train_data, train_label)

    p = clf.predict(train_data)
    train_acc = accuracy_score(train_label, p)

    p = clf.predict(test_data)
    test_acc = accuracy_score(test_label, p)

    return train_acc, test_acc


class cca_loss():

    def __init__(self, outdim_size, use_all_singular_values):
        self.outdim_size = outdim_size
        self.use_all_singular_values = use_all_singular_values

    def loss(self, H1, H2):
        """
        It is the loss function of CCA as introduced in the original paper. There can be other formulations.
        use_all_singular_values只能设置为False，如果是True的话梯度会出现nan，原因不明
        有许多assert语句，这是为了保证参与计算的矩阵里面没有nan，不然会有各种奇怪的错误
        """

        r1 = 1e-3
        r2 = 1e-3
        eps = 1e-9

        # print(H1.size())
        # print(H2.size())

        # 输入的H1和H2是 sample x feature
        H1, H2 = H1.t(), H2.t()
        # 这里转置后H1和H2是 feature x sample

        assert torch.isnan(H1).sum().item() == 0
        assert torch.isnan(H2).sum().item() == 0

        # o1和o2是feature的数量
        o1 = H1.size(0)
        o2 = H2.size(0)

        m = H1.size(1)  # m是sample数量

        # print(H1.size())

        # 减去自身均值，减完之后，H1bar，H2bar的均值皆为0，这样后面好求协方差矩阵
        H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
        H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)

        assert torch.isnan(H1bar).sum().item() == 0
        assert torch.isnan(H2bar).sum().item() == 0

        # 求协方差矩阵
        SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())
        SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(H1bar,
                                                    H1bar.t()) + r1 * torch.eye(o1)
        SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2bar,
                                                    H2bar.t()) + r2 * torch.eye(o2)

        assert torch.isnan(SigmaHat11).sum().item() == 0
        assert torch.isnan(SigmaHat12).sum().item() == 0
        assert torch.isnan(SigmaHat22).sum().item() == 0

        # Calculating the root inverse of covariance matrices by using eigen decomposition
        # 特征分解，得到特征值和特征向量

        [D1, V1] = torch.linalg.eigh(SigmaHat11)
        [D2, V2] = torch.linalg.eigh(SigmaHat22)

        assert torch.isnan(D1).sum().item() == 0
        assert torch.isnan(D2).sum().item() == 0
        assert torch.isnan(V1).sum().item() == 0
        assert torch.isnan(V2).sum().item() == 0

        # Added to increase stability
        # 取非0元素
        posInd1 = torch.gt(D1, eps).nonzero()[:, 0]
        D1 = D1[posInd1]
        V1 = V1[:, posInd1]
        posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
        D2 = D2[posInd2]
        V2 = V2[:, posInd2]
        # print(posInd1.size())
        # print(posInd2.size())

        # 求协方差矩阵11和22的-1/2次方
        SigmaHat11RootInv = torch.matmul(
            torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
        SigmaHat22RootInv = torch.matmul(
            torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())

        Tval = torch.matmul(torch.matmul(SigmaHat11RootInv,
                                         SigmaHat12), SigmaHat22RootInv)

        # print(Tval.size())

        if self.use_all_singular_values:
            # all singular values are used to calculate the correlation
            tmp = torch.matmul(Tval.t(), Tval)
            corr = torch.trace(torch.sqrt(tmp))
            # assert torch.isnan(corr).item() == 0
        else:
            # just the top self.outdim_size singular values are used
            trace_TT = torch.matmul(Tval.t(), Tval)
            trace_TT = torch.add(trace_TT, (torch.eye(trace_TT.shape[0])*r1))  # regularization for more stability
            U, V = torch.linalg.eigh(trace_TT)
            U = torch.where(U > eps, U, (torch.ones(U.shape).float()*eps))
            U = U.topk(self.outdim_size)[0]
            corr = torch.sum(torch.sqrt(U))

        return -corr


class DeepCCA_EEGNet(nn.Module):

    def __init__(self, args_eeg, args_nirs):

        super(DeepCCA_EEGNet, self).__init__()

        self.feature_eeg = EEGNet.EEGNet_eeg(n_classes=2, Chans=args_eeg['channel'], Samples=args_eeg['time'], kernLength=args_eeg['kernlength'])
        self.feature_nirs = EEGNet.EEGNet_nirs(n_classes=2, Chans=args_nirs['channel'], Samples=args_nirs['time'], kernLength=args_nirs['kernlength'])

    def forward(self, eeg, nirs):

        out_eeg = self.feature_eeg(eeg)
        out_nirs = self.feature_nirs(nirs)

        return out_eeg, out_nirs


class DeepCCA_CRNN(nn.Module):

    def __init__(self, args_eeg, args_nirs):

        super(DeepCCA_CRNN, self).__init__()

        self.feature_eeg = CRNN.CRNN_eeg()
        self.feature_nirs = CRNN.CRNN_nirs()

    def forward(self, eeg, nirs):

        out_eeg = self.feature_eeg(eeg)
        out_nirs = self.feature_nirs(nirs)

        return out_eeg, out_nirs


def Train(train_dataset, test_dataset, base):
    # 超参数
    args_EEGNet = {
        'eeg': {'channel': 30,
                'time': 400,
                'kernlength': 100,
                'out_dim': 16 * 400 // 100
                },
        'nirs': {'channel': 36,
                 'time': 20,
                 'kernlength': 10,
                 'out_dim': 16 * 20 // 10
                 },
        'cca': {'eeg_dim': 16 * 400 // 100,
                'nirs_dim': 16 * 20 // 10,
                'out_dim': 8,
                'all_singular': False
                }
    }

    args_CRNN = {
        'eeg': {'channel': (15, 15),
                'time': 400,
                'out_dim': 16
                },
        'nirs': {'channel': (15, 15),
                 'time': 20,
                 'out_dim': 16
                 },
        'cca': {'eeg_dim': 16,
                'nirs_dim': 16,
                'out_dim': 8,
                'all_singular': False
                }
    }

    train_batch_size = 20
    batch_print = 6
    test_batch_size = 20

    # 最后加权融合时两种view数据的权重
    a = 0.6

    # 读取数据集
    train_loader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=True)

    print("train_samples: %d" % train_dataset.__len__())
    print("test_samples: %d" % test_dataset.__len__())
    print("\n")

    # 给网络实例化
    if base == 'EEGNet':
        args = args_EEGNet
        net = DeepCCA_EEGNet(args['eeg'], args['nirs'])
        print(net)
    else:
        args = args_CRNN
        net = DeepCCA_CRNN(args['eeg'], args['nirs'])
        print(net)

    # 参数初始化
    if base == 'EEGNet':
        net.apply(EEGNet.weights_init)
    else:
        net.apply(CRNN.weights_init)

    # 定义损失函数
    cca = cca_loss(outdim_size=args['cca']['out_dim'], use_all_singular_values=args['cca']['all_singular'])

    # 定义更新网络参数的算法（这里用随机梯度下降法）
    optimizer = optim.RMSprop(net.parameters(), lr=0.00001, weight_decay=1e-3)
    # optimizer = optim.LBFGS(net.parameters(), lr=1)
    # optimizer = optim.Adam(net.parameters(), lr=0.0000001, weight_decay=1e-3)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.8, last_epoch=-1)

    # 训练
    train_epoch = 200

    # 画图用
    X_epoch = []
    Y_acc_train = []
    Y_acc_test = []

    for epoch in range(train_epoch):

        train_loss = 0.0
        finish = False

        print("learning rate:", optimizer.state_dict()['param_groups'][0]['lr'])

        # 训练
        net.train()

        for index, data in enumerate(train_loader):

            input_eeg, input_nirs, label = data
            # 调整一下数据类型，不然会报错
            input_eeg = input_eeg.to(torch.float32)
            input_nirs = input_nirs.to(torch.float32)
            label = label.to(torch.long)

            # print(input_eeg.shape)
            # print(input_nirs.shape)
            # print(label.shape)

            '''
            def closure():

                optimizer.zero_grad()  # 梯度置0
                output1, output2 = net(input_eeg, input_nirs)  # 前向传播
                loss = cca.loss(output1, output2)   # 计算cca损失
                loss.backward()  # 反向传播计算梯度
                return loss
            '''

            optimizer.zero_grad()  # 梯度置0

            output1, output2 = net(input_eeg, input_nirs)  # 前向传播

            loss = cca.loss(output1, output2)

            loss.backward()  # 反向传播计算梯度

            optimizer.step()  # 更新参数

            train_loss += loss.item()

            if index % batch_print == batch_print - 1:  # 每n0个batch打印一次结果，并检查损失是否小于阈值

                print("[%d, %d] loss: %.3f" % (epoch + 1, index + 1, train_loss / batch_print))

                # if train_loss / batch_print < 0.05:
                #     finish = True
                #     break
                train_loss = 0.0

        if finish:
            break

        # 测试
        net.eval()

        with torch.no_grad():

            def forward(loader):

                x = []
                y = []
                loss = 0.0

                for index, data in enumerate(loader):

                    input_eeg, input_nirs, label = data

                    # 调整一下数据类型，不然会报错
                    input_eeg = input_eeg.to(torch.float32)
                    input_nirs = input_nirs.to(torch.float32)
                    label = label.to(torch.long)

                    output1, output2 = net(input_eeg, input_nirs)

                    loss = cca.loss(output1, output2)

                    loss += loss.item()

                    output = a * output1 + (1 - a) * output2

                    x.append(output)
                    y.append(label)

                feature_clf = torch.cat(x, dim=0).numpy()
                label_clf = torch.cat(y, dim=0).numpy()

                return loss, feature_clf, label_clf

            train_loss, feature_train, label_train = forward(train_loader)
            test_loss, feature_test, label_test = forward(test_loader)

        # 用SVM作为分类器
        train_acc, test_acc = svm_classify(feature_train, label_train, feature_test, label_test)

        # 保存画图数据
        X_epoch.append(epoch + 1)
        Y_acc_train.append(train_acc)
        Y_acc_test.append(test_acc)

        print("train loss: %.3f  train accuracy: %.2f%%  test loss: %.3f  test accuracy: %.2f%% \n"
              % (train_loss / (train_dataset.__len__() // train_batch_size), train_acc,
                 test_loss / (test_dataset.__len__() // test_batch_size), test_acc))

        scheduler.step()

    return X_epoch, Y_acc_train, Y_acc_test