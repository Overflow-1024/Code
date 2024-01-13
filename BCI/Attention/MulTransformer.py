import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary

import math
import Config as cfg

from Attention import Transformer as tf
from Hybrid import LMF


class MultimodalEncoder(nn.Module):

    def __init__(self, emb_dim, head, ff_dim, size, drop):
        super(MultimodalEncoder, self).__init__()

        self.x_norm1 = tf.LayerNorm(size=size)
        self.x_self_attn = tf.MultiHeadAttention(emb_dim=emb_dim, head=head)
        self.x_dropout1 = nn.Dropout(drop)

        self.x_norm2 = tf.LayerNorm(size=size)
        self.x_feed_forward = tf.FeedForward(d_model=emb_dim, d_ff=ff_dim, drop=drop)
        self.x_dropout2 = nn.Dropout(drop)

        self.y_norm1 = tf.LayerNorm(size=size)
        self.y_self_attn = tf.MultiHeadAttention(emb_dim=emb_dim, head=head)
        self.y_dropout1 = nn.Dropout(drop)

        self.y_norm2 = tf.LayerNorm(size=size)
        self.y_feed_forward = tf.FeedForward(d_model=emb_dim, d_ff=ff_dim, drop=drop)
        self.y_dropout2 = nn.Dropout(drop)

    def forward(self, x, y):

        x = self.x_norm1(x)
        y = self.y_norm1(y)

        x_att = self.x_self_attn(x, y, y)
        x_att = self.x_dropout1(x_att)

        y_att = self.y_self_attn(y, x, x)
        y_att = self.y_dropout1(y_att)

        x = x + x_att
        y = y + y_att

        x = self.x_norm2(x)
        y = self.y_norm2(x)

        x_ff = self.x_feed_forward(x)
        x_ff = self.x_dropout2(x_ff)

        y_ff = self.y_feed_forward(y)
        y_ff = self.y_dropout2(y_ff)

        x = x + x_ff
        y = y + y_ff

        return x, y


class MultimodalDecoder(nn.Module):

    def __init__(self, emb_dim, seq_len):

        super(MultimodalDecoder, self).__init__()

        self.conv_x = nn.Conv1d(in_channels=emb_dim, out_channels=emb_dim, kernel_size=seq_len)

        self.conv_y = nn.Conv1d(in_channels=emb_dim, out_channels=emb_dim, kernel_size=seq_len)

        self.hybrid = LMF.LMF(eeg_dim=emb_dim, nirs_dim=emb_dim, out_dim=emb_dim, rank=4)

        self.fc = nn.Linear(in_features=emb_dim, out_features=2)

    def forward(self, x, y):

        x = self.conv_x(x)
        x = x.squeeze()

        y = self.conv_y(y)
        y = y.squeeze()

        out = self.hybrid(x, y)
        out = self.fc(out)

        return out


class MultimodalTransformer(nn.Module):

    def __init__(self, args_att, args_eeg, args_nirs):
        super(MultimodalTransformer, self).__init__()

        self.emb_eeg = tf.Embeddings(in_channel=args_eeg['in_channel'], channel=args_eeg['channel'], F1=args_eeg['F1'],
                              emb_dim=args_eeg['emb_dim'], kern_length=args_eeg['kern_length'], drop=args_eeg['drop'], data_type=args_eeg['data_type'])
        self.pos_eeg = tf.PositionalEncoding(d_model=args_eeg['emb_dim'], drop=args_eeg['drop'])

        self.emb_nirs = tf.Embeddings(in_channel=args_nirs['in_channel'], channel=args_nirs['channel'], F1=args_nirs['F1'],
                              emb_dim=args_nirs['emb_dim'], kern_length=args_nirs['kern_length'], drop=args_nirs['drop'], data_type=args_nirs['data_type'])
        self.pos_nirs = tf.PositionalEncoding(d_model=args_nirs['emb_dim'], drop=args_nirs['drop'])

        self.encoder = MultimodalEncoder(emb_dim=args_att['emb_dim'], head=args_att['head'], ff_dim=args_att['ff_dim'], size=args_att['size'], drop=args_att['drop'])

        self.decoder = MultimodalDecoder(emb_dim=args_att['emb_dim'], seq_len=args_att['seq_len'])

    def forward(self, x, y):

        # Size here: (batch, F1, channel, time)
        x = self.emb_eeg(x)
        x = x.transpose(-1, -2)
        x = self.pos_eeg(x)

        y = self.emb_nirs(y)
        y = y.transpose(-1, -2)
        y = self.pos_nirs(y)
        # Size here: (batch, seq_len, emb_dim)

        x, y = self.encoder(x, y)

        # Size here: (batch, seq_len, emb_dim)

        x = x.transpose(-1, -2)
        y = y.transpose(-1, -2)
        # Size here: (batch, emb_dim, seq_len)

        out = self.decoder(x, y)

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


def Train(train_dataset, test_dataset):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 读取参数
    config_MulT = cfg.MulTransformer

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
    eeg_args = config_MulT['eeg_args']
    nirs_args = config_MulT['nirs_args']
    att_args = config_MulT['att_args']
    hyper_param = config_MulT['hyper_param']

    net = MultimodalTransformer(att_args, eeg_args, nirs_args)
    print(net)

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

            input_eeg, input_nirs, label = data

            # 调整一下数据类型，不然会报错
            input_eeg = input_eeg.to(torch.float32)
            input_nirs = input_nirs.to(torch.float32)
            label = label.to(torch.long)

            input_eeg = input_eeg.to(device)
            input_nirs = input_nirs.to(device)
            label = label.to(device)

            # print(input.shape)
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


