from torch import nn
from torch.nn import Module
import torch.nn.functional as F
import brevitas.nn as qnn


class QuantWeightLeNet(Module):
    def __init__(self):
        super(QuantWeightLeNet, self).__init__()
        self.conv1 = qnn.QuantConv2d(3, 6, 5, bias=True, weight_bit_width=4)
        self.relu1 = nn.ReLU()
        self.conv2 = qnn.QuantConv2d(6, 16, 5, bias=True, weight_bit_width=4)
        self.relu2 = nn.ReLU()
        self.fc1 = qnn.QuantLinear(16 * 5 * 5, 120, bias=True, weight_bit_width=4)
        self.relu3 = nn.ReLU()
        self.fc2 = qnn.QuantLinear(120, 84, bias=True, weight_bit_width=4)
        self.relu4 = nn.ReLU()
        self.fc3 = qnn.QuantLinear(84, 10, bias=True, weight_bit_width=4)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = self.relu2(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.shape[0], -1)
        out = self.relu3(self.fc1(out))
        out = self.relu4(self.fc2(out))
        out = self.fc3(out)
        return out
