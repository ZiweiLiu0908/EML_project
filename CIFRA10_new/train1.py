import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torch.nn as nn
from brevitas.export import export_onnx_qcdq

from model1 import QuantWeightLeNet

import torch.optim as optim
import torchvision.transforms as transforms

weight_bit_width = None


def train():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,
                                               shuffle=True, num_workers=0)

    val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=36,
                                             shuffle=False, num_workers=0)
    val_data_iter = iter(val_loader)
    val_image, val_label = next(val_data_iter)
    global weight_bit_width
    net = QuantWeightLeNet(weight_bit_width)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    Loss = []
    t = []
    Acc = []
    max_acc = 0
    f = open('./exp1.txt', "w")
    for epoch in range(30):
        for step, data in enumerate(train_loader, start=0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            t.append(loss.item())
        Loss.append(sum(t) / len(t))
        t = []

        with torch.no_grad():
            outputs = net(val_image)
            predict_y = torch.max(outputs, dim=1)[1]
            accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)
            Acc.append(accuracy)
            print('weight_bit_width：%d epoch: %d train_loss: %.3f  test_accuracy: %.3f' %
                  (weight_bit_width, epoch + 1, Loss[-1], accuracy))
            f.write('weight_bit_width：%d epoch: %d train_loss: %.3f  test_accuracy: %.3f' %
                  (weight_bit_width, epoch + 1, Loss[-1], accuracy))
            if accuracy > max_acc:
                max_acc = max(max_acc, accuracy)
                try:

                    export_onnx_qcdq(net, torch.randn(36, 3, 32, 32),
                                     export_path='./onnx/model1_%d_{}.onnx' % weight_bit_width)
                    print(f"save model   ./onnx/model1_{weight_bit_width}.onnx     successfully")
                except:
                    print("save model failed")

    print('Finished Training')

    Loss = np.array(Loss)
    Acc = np.array(Acc)

    plt.xlabel("epoch")
    plt.plot(Loss, label="loss")
    plt.legend()
    plt.savefig("./statistics/model1_loss_{%d}_{ }" % weight_bit_width)
    plt.close()

    plt.xlabel("epoch")
    plt.plot(Acc, label="accuracy")
    plt.legend()
    plt.savefig("./statistics/model1_accuracy_{%d}_{ }" % weight_bit_width)
    plt.close()


def main():
    global weight_bit_width
    for i in range(2, 9):
        weight_bit_width = i
        train()
        print(
            "##############################################################################################################################")


if __name__ == '__main__':
    main()
