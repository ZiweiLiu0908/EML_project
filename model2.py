from torch.nn import Module
import torch.nn.functional as F

import brevitas.nn as qnn
from brevitas.quant import Int8Bias as BiasQuant


class QuantWeightActLeNet(Module):
    """
        Weights and activations quantization, float biases
    """

    def __init__(self):
        super(QuantWeightActLeNet, self).__init__()
        self.quant_inp = qnn.QuantIdentity(bit_width=4)
        self.conv1 = qnn.QuantConv2d(1, 6, 5, bias=True, weight_bit_width=4)
        self.relu1 = qnn.QuantReLU(bit_width=4)
        self.conv2 = qnn.QuantConv2d(6, 16, 5, bias=True, weight_bit_width=4)
        self.relu2 = qnn.QuantReLU(bit_width=3)
        self.fc1 = qnn.QuantLinear(256, 120, bias=True, weight_bit_width=4)
        self.relu3 = qnn.QuantReLU(bit_width=4)
        self.fc2 = qnn.QuantLinear(120, 84, bias=True, weight_bit_width=4)
        self.relu4 = qnn.QuantReLU(bit_width=4)
        self.fc3 = qnn.QuantLinear(84, 10, bias=True)

    def forward(self, x):
        out = self.quant_inp(x)
        out = self.relu1(self.conv1(out))
        out = F.max_pool2d(out, 2)
        out = self.relu2(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.reshape(out.shape[0], -1)
        out = self.relu3(self.fc1(out))
        out = self.relu4(self.fc2(out))
        out = self.fc3(out)
        return out
