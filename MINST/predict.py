import torch
import torchvision.transforms as transforms
from PIL import Image

from model_0 import Model


def main():
    transform = transforms.Compose(
      [transforms.ToTensor()]
    )


    classes = ('0', '1', '2', '3',
               '4', '5', '6', '7', '8', '9')

    net = Model()
    net.load_state_dict(torch.load('models_0/mnist_0.884.onnx'))

    im = Image.open('1.jpg')
    im = transform(im)  # [C, H, W]
    im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]

    with torch.no_grad():
        outputs = net(im)
        predict = torch.max(outputs, dim=1)[1].numpy()
    print(classes[int(predict)])


if __name__ == '__main__':
    main()