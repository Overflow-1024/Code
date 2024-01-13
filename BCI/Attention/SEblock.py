import torch
import torch.nn as nn
import torch.nn.functional as func


class SEblock(nn.Module):

    # ratio是压缩参数
    def __init__(self, channels, ratio):

        super(SEblock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(in_channels=channels, out_channels=channels//ratio, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.excitation = nn.Conv2d(in_channels=channels//ratio, out_channels=channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

    def forward(self, x):

        out = self.squeeze(x)
        out = self.compress(out)
        out = func.relu(out)
        out = self.excitation(out)
        out = torch.sigmoid(out)

        return out
