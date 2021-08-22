import torch
import torchvision.transforms as transforms
from PIL import Image

from models.LeNet.lenet_torch import LeNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    net = LeNet().to(device)
    net.load_state_dict(torch.load('./weights/Lenet.pth'))
    im = Image.open('./data/1.jpg')
    im = transform(im).to(device)  # [C, H, W]
    im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]

    with torch.no_grad():
        outputs = net(im).cpu()
        predict = torch.max(outputs, dim=1)[1].data.numpy()
    print(classes[int(predict)])


if __name__ == '__main__':
    main()
