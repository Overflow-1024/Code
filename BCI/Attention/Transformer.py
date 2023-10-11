import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary

import math

import Config as cfg


class SelfAttention(nn.Module):

    def __init__(self, emb_dim, model_dim):

        super(SelfAttention, self).__init__()

        self.embDim = emb_dim
        self.modelDim = model_dim

        self.calc_query = nn.Linear(in_features=self.embDim, out_features=self.modelDim)
        self.calc_key = nn.Linear(in_features=self.embDim, out_features=self.modelDim)
        self.calc_value = nn.Linear(in_features=self.embDim, out_features=self.modelDim)

    def forward(self, q, k, v):

        # 输入x维度(3): batch × time × emb_dim(channel)

        # 对输入做线性变换，得到query，key，value
        query = self.calc_query(q)
        key = self.calc_key(k)
        value = self.calc_value(v)

        d_k = self.modelDim
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        p_attn = scores.softmax(dim=-1)
        result = torch.matmul(p_attn, value)

        return result


class MultiHeadAttention(nn.Module):

    def __init__(self, emb_dim, head):
        super(MultiHeadAttention, self).__init__()

        self.embDim = emb_dim
        assert emb_dim % head == 0
        self.h = head
        self.modelDim = emb_dim // head

        self.calc_query = nn.Linear(in_features=self.embDim, out_features=self.embDim)
        self.calc_key = nn.Linear(in_features=self.embDim, out_features=self.embDim)
        self.calc_value = nn.Linear(in_features=self.embDim, out_features=self.embDim)

        self.calc_out = nn.Linear(in_features=self.embDim, out_features=self.embDim)

    def forward(self, q, k, v):

        # 输入x维度(3): batch × time × emb_dim(channel)
        batch = q.size(0)

        # 对输入做线性变换，得到query，key，value
        query = self.calc_query(q)
        key = self.calc_key(k)
        value = self.calc_value(v)

        # 将 emb_dim 维度拆解成 h × model_dim
        query = query.view(batch, -1, self.h, self.modelDim).transpose(1, 2)
        key = key.view(batch, -1, self.h, self.modelDim).transpose(1, 2)
        value = value.view(batch, -1, self.h, self.modelDim).transpose(1, 2)

        # Apply attention on all the projected vectors in batch
        d_k = self.modelDim
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        p_attn = scores.softmax(dim=-1)
        result = torch.matmul(p_attn, value)

        # "Concat" using a view and apply a final linear.
        out = result.transpose(1, 2).contiguous().view(batch, -1, self.h * self.modelDim)
        out = self.calc_out(out)

        return out


class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff, drop):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(in_features=d_model, out_features=d_ff)
        self.w_2 = nn.Linear(in_features=d_ff, out_features=d_model)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


class LayerNorm(nn.Module):

    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))
        self.eps = eps

        nn.init.constant_(self.a_2, 1)
        nn.init.constant_(self.b_2, 1)

    def forward(self, x):

        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, drop, max_len=2000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=drop)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):

        x = x + self.pe[:, : x.size(1)].requires_grad_(False)

        return self.dropout(x)


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)


class Embeddings(nn.Module):

    def __init__(self, in_channel, channel, F1, emb_dim, kern_length, drop, data_type):
        super(Embeddings, self).__init__()

        self.Channel = channel
        self.kernLength = kern_length

        self.F1 = F1
        self.embDim = emb_dim
        self.dataType = data_type

        self.conv_temporal = nn.Conv2d(in_channels=in_channel, out_channels=self.F1, kernel_size=(1, self.kernLength), stride=(1, 1), padding=(0, (self.kernLength-1)//2), bias=False)
        self.bn_temporal = nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3)
        self.conv_spatial = Conv2dWithConstraint(in_channels=self.F1, out_channels=self.embDim, kernel_size=(self.Channel, 1), stride=(1, 1), padding=(0, 0), groups=self.F1, bias=False, max_norm=1)
        self.bn_spatial = nn.BatchNorm2d(self.embDim, momentum=0.01, affine=True, eps=1e-3)

        self.pool = nn.AvgPool2d(kernel_size=(1, 20), stride=(1, 20))   # 200/10=20Hz
        self.dropout = nn.Dropout(p=drop)

    def forward(self, x):

        # Size here: (batch, 1, channel, time)
        x = self.conv_temporal(x)
        x = self.bn_temporal(x)
        # Size here: (batch, F1, channel, time)
        x = self.conv_spatial(x)
        x = self.bn_spatial(x)
        # Size here: (batch, emb_dim, 1, time)
        x = func.elu(x)
        if self.dataType == 'eeg':
            x = self.pool(x)
        x = self.dropout(x)

        out = x.squeeze()
        # Size here: (batch, seq_len, emb_dim)

        return out


class Encoder(nn.Module):

    def __init__(self, emb_dim, head, ff_dim, size, drop):
        super(Encoder, self).__init__()

        self.norm1 = LayerNorm(size=size)
        self.self_attn = MultiHeadAttention(emb_dim=emb_dim, head=head)
        self.dropout1 = nn.Dropout(drop)

        self.norm2 = LayerNorm(size=size)
        self.feed_forward = FeedForward(d_model=emb_dim, d_ff=ff_dim, drop=drop)
        self.dropout2 = nn.Dropout(drop)

    def forward(self, x):

        x = self.norm1(x)
        x_att = self.self_attn(x, x, x)
        x_att = self.dropout1(x_att)

        x = x + x_att

        x = self.norm2(x)
        x_ff = self.feed_forward(x)
        x_ff = self.dropout2(x_ff)

        out = x + x_ff

        return out


class Decoder(nn.Module):

    def __init__(self, emb_dim, seq_len):
        super(Decoder, self).__init__()

        self.conv = nn.Conv1d(in_channels=emb_dim, out_channels=emb_dim // 2, kernel_size=seq_len)
        self.fc = nn.Linear(in_features=emb_dim // 2, out_features=2)

    def forward(self, x):

        x = self.conv(x)
        x = x.squeeze()
        out = self.fc(x)

        return out


class Transformer(nn.Module):

    def __init__(self, args_att, args):
        super(Transformer, self).__init__()

        self.emb = Embeddings(in_channel=args['in_channel'], channel=args['channel'], F1=args['F1'],
                              emb_dim=args['emb_dim'], kern_length=args['kern_length'], drop=args['drop'], data_type=args['data_type'])
        self.pos = PositionalEncoding(d_model=args['emb_dim'], drop=args['drop'])

        self.encoder = Encoder(emb_dim=args_att['emb_dim'], head=args_att['head'], ff_dim=args_att['ff_dim'], size=args_att['size'], drop=args_att['drop'])

        self.decoder = Decoder(emb_dim=args_att['emb_dim'], seq_len=args_att['seq_len'])

    def forward(self, x):

        # Size here: (batch, F1, channel, time)
        x = self.emb(x)
        x = x.transpose(-1, -2)
        # Size here: (batch, seq_len, emb_dim)

        x = self.pos(x)
        # Size here: (batch, seq_len, emb_dim)

        x = self.encoder(x)
        # Size here: (batch, seq_len, emb_dim)

        x = x.transpose(-1, -2)
        # Size here: (batch, emb_dim, seq_len)

        out = self.decoder(x)

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
    config_eeg = cfg.Transformer_eeg
    config_nirs = cfg.Transformer_nirs

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
        eeg_args = config_eeg['eeg_args']
        att_args = config_eeg['att_args']
        hyper_param = config_eeg['hyper_param']

        net = Transformer(att_args, eeg_args)
        print(net)
        summary(net, input_size=(1, eeg_args['channel'], eeg_args['time']))
    else:
        nirs_args = config_nirs['nirs_args']
        att_args = config_nirs['att_args']
        hyper_param = config_nirs['hyper_param']

        net = Transformer(att_args, nirs_args)
        print(net)
        summary(net, input_size=(2, nirs_args['channel'], nirs_args['time']))

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
