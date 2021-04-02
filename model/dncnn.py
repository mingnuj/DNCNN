import torch.nn as nn


class DNCNN(nn.Module):
    def __init__(self, in_planes=3, blocks=17, hidden=64, kernel_size=3, padding=0, bias=False):
        super(DNCNN, self).__init__()
        self.conv_f = nn.Conv2d(in_channels=in_planes, out_channels=hidden, kernel_size=kernel_size, padding=padding, bias=bias)
        self.conv_h = nn.Conv2d(in_channels=hidden, out_channels=hidden, kernel_size=kernel_size, padding=padding, bias=bias)
        self.conv_l = nn.Conv2d(in_channels=hidden, out_channels=in_planes, kernel_size=kernel_size, padding=padding, bias=bias)

        self.bn = nn.BatchNorm2d(hidden)
        self.relu = nn.ReLU(inplace=True)

        self.hidden_layer = self.mk_hidden_layer(blocks)

    def mk_hidden_layer(self, blocks=17):
        layers = []
        for _ in range(blocks-2):
            layers.append(self.conv_h)
            layers.append(self.bn)
            layers.append(self.relu)
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv_f(x)
        out = self.relu(out)

        out = self.hidden_layer(out)

        out = self.conv_l(out)

        return out
