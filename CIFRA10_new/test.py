import os
import time
import torch
import shutil
import psutil
import argparse
import torchvision
import torch.nn as nn
import onnxruntime as ort
import torch.optim as optim
import torchvision.transforms as transforms



f = open('./test_result.txt', "a")
def test(opt):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=36,
                                             shuffle=False, num_workers=0)
    val_data_iter = iter(val_loader)
    val_image, val_label = next(val_data_iter)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    start_time = time.time()

    # opt.name = opt.name + ".onnx"
    ort_session = ort.InferenceSession(opt.name)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(val_image)}
    ort_outs = ort_session.run(None, ort_inputs)
    if len(ort_outs) > 1:
        ort_outs = ort_outs[0]
    ort_outs = torch.tensor(ort_outs).squeeze()
    predict_y = torch.max(ort_outs, dim=1)[1]
    accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)

    run_time = time.time() - start_time

    print("test model:{}".format(opt.name))
    print("result:run time[{}],accuracy[{}]".format(run_time, accuracy))
    f.write("test model:{}  ".format(opt.name) +  "  result:run time[{}],accuracy[{}]\n".format(run_time, accuracy))
    f.close()


def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default="test", help='train/test')
    parser.add_argument('--epoch', type=int, default=10, help='epoch num')
    parser.add_argument('--name', type=str, default="./onnx/model3_2_2.onnx", help='save model name')
    # parser.add_argument('--model_choose', type=int, default=3, help='choose model')
    parser.add_argument('--vis', action="store_true", help='save model name')

    return parser.parse_args()


# python train0.py --mode train --name model1 --model_choose 1 --vis
# python train0.py --mode test --name model1
if __name__ == '__main__':
    opt = parse_opt()
    test(opt)