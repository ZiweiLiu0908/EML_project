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

from memory_profiler import profile
from brevitas.export import export_onnx_qcdq

import model_0
import model1
import model2
import model3

summaryWriter=''

def record_memory():
    process = psutil.Process()  # 获取当前进程
    memory_usage = process.memory_info().rss  # 获取当前进程的内存占用（字节）
    return memory_usage

def train(opt):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,
                                               shuffle=True, num_workers=0)

    val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=5000,
                                             shuffle=False, num_workers=0)
    val_data_iter = iter(val_loader)
    val_image, val_label = next(val_data_iter)

    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    if opt.model_choose==0:
        net=model_0.LeNet()
    elif opt.model_choose==1:
        net=model1.QuantWeightLeNet()
    elif opt.model_choose==2:
        net=model2.QuantWeightLeNet()
    else:
        net=model3.QuantWeightLeNet()

    if opt.vis==True:
        from torch.utils.tensorboard import SummaryWriter
        if os.path.exists(opt.name):  # 判断文件夹是否存在
            shutil.rmtree(opt.name)  # 删除文件夹及其内容
        global summaryWriter
        summaryWriter = SummaryWriter(opt.name)
        

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    start_time = time.time()
    for epoch in range(opt.epoch):  # loop over the dataset multiple times

        running_loss = 0.0
        for step, data in enumerate(train_loader, start=0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data   
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if step % 500 == 499:  # print every 500 mini-batches
                if epoch==0:
                    print("model {}".format(opt.model_choose))
                print("epoch {} - iteration {}: average train loss {:.3f}".format(epoch+1, step, running_loss/500))

                with torch.no_grad():
                    outputs = net(val_image)
                    testing_loss = loss_function(outputs, val_label).item()
                print("epoch {} - iteration {}: average test loss {:.3f}".format(epoch+1, step, testing_loss))

                if opt.vis==True:
                    summaryWriter.add_graph(net,inputs)
                    summaryWriter.add_scalar("{} training_loss".format(opt.name),running_loss/500, epoch * len(train_loader) + step)
                    summaryWriter.add_scalar("{} testing_loss".format(opt.name),testing_loss, epoch * len(train_loader) + step)
                    memory_usage = record_memory() 
                    summaryWriter.add_scalar("{}memory_usage".format(opt.name),memory_usage,epoch* len(train_loader) + step)
                    run_time = time.time() - start_time
                    summaryWriter.add_scalar("{} train_time".format(opt.name),run_time,epoch* len(train_loader) + step)

                running_loss = 0.0             

    print('Finished Training')

    ##########################################################
    opt.name=opt.name+'.onnx'
    dynamic_axes_0 = { 
    'in' : [0], 
    'out' : [0] 
    } 
    torch.onnx.export(net, torch.randn(36, 3, 32, 32), opt.name,input_names=['in'], output_names=['out'], dynamic_axes=dynamic_axes_0)
    ##########################################################

@profile(precision=4)
def test(opt):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=5000,
                                             shuffle=False, num_workers=0)
    val_data_iter = iter(val_loader)
    val_image, val_label = next(val_data_iter)

    
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    start_time = time.time()

    opt.name=opt.name+".onnx"
    ort_session = ort.InferenceSession(opt.name)
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(val_image)}
    ort_outs = torch.tensor(ort_session.run(None, ort_inputs)).squeeze()
    predict_y = torch.max(ort_outs, dim=1)[1]
    accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)

    run_time = time.time() - start_time

    print("test model:{}".format(opt.name))
    print("result:run time[{}],accuracy[{}]".format(run_time,accuracy))


def main(opt):
    if opt.mode=='train':
        train(opt)
    else:
        test(opt)

def parse_opt():
    parser = argparse.ArgumentParser()
     
    parser.add_argument('--mode', type=str, default="train", help='train/test')
    parser.add_argument('--epoch', type=int, default=10, help='epoch num')
    parser.add_argument('--name', type=str, default="mode.onnx", help='save model name')
    parser.add_argument('--model_choose', type=int, default=1, help='choose model')
    parser.add_argument('--vis', action="store_true",  help='save model name')

    return parser.parse_args()

#python main.py --mode train --name model1 --model_choose 1 --vis
#python main.py --mode test --name model1
if __name__ == '__main__':
    opt = parse_opt()
    main(opt)