from torch.nn import Module
from torch import nn
import brevitas.nn as qnn
from brevitas.quant import Int8Bias as BiasQuant


class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.quant_inp = qnn.QuantIdentity(bit_width=4, return_quant_tensor=True)

        self.conv1 = qnn.QuantConv2d(1, 6, 5, weight_bit_width=3, bias_quant=BiasQuant, return_quant_tensor=True)
        self.relu1 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)
        self.pool1 = qnn.QuantMaxPool2d(2)
        self.conv2 = qnn.QuantConv2d(6, 16, 5, weight_bit_width=3, bias_quant=BiasQuant, return_quant_tensor=True)
        self.relu2 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)
        self.pool2 = qnn.QuantMaxPool2d(2)
        self.fc1 = qnn.QuantLinear(256, 120, bias=True, weight_bit_width=3, bias_quant=BiasQuant, return_quant_tensor=True)
        self.relu3 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)
        self.fc2 = qnn.QuantLinear(120, 84, bias=True, weight_bit_width=3, bias_quant=BiasQuant, return_quant_tensor=True)
        self.relu4 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)
        self.fc3 = qnn.QuantLinear(84, 10, bias=False, weight_bit_width=3, bias_quant=BiasQuant, return_quant_tensor=True)
        self.relu5 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)

    def forward(self, x):
        y = self.quant_inp(x)
        y = self.conv1(y)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        return y