import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

from models.LeNet.lenet_torch import LeNet


def main(show=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # 50000张训练图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=36, shuffle=True, num_workers=0)
    # 10000张验证图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    val_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    if show:
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=4, shuffle=False, num_workers=0)
        val_data_iter = iter(val_loader)
        val_image, val_label = val_data_iter.next()
        def imshow(img):
            img = img/2+0.5
            npimg = img.numpy()
            plt.imshow(np.transpose(npimg,(1,2,0)))
            plt.show()
        print(" ".join('%5s'%classes[val_label[j]] for j in range(4)))
        imshow(torchvision.utils.make_grid(val_image))
        return
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=5000, shuffle=False, num_workers=0)
    val_data_iter = iter(val_loader)
    val_image, val_label = val_data_iter.next()

    net = LeNet().to(device)
    loss_function = nn.CrossEntropyLoss() # 损失函数 already contains softmax and NLLLoss
    optimizer = optim.Adam(net.parameters(), lr=0.001) # 优化器

    for epoch in range(5):  # loop over the dataset multiple times
        running_loss = 0.0
        for step, data in enumerate(train_loader, start=0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if step % 500 == 499:  # print every 500 mini-batches
                with torch.no_grad():
                    outputs = net(val_image.to(device))  # [batch, 10]
                    predict_y = torch.max(outputs, dim=1)[1]
                    accuracy = torch.eq(predict_y, val_label.to(device)).sum().item() / val_label.size(0)
                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy))
                    running_loss = 0.0
    print('Finished Training')
    save_path = './weights/Lenet.pth'
    torch.save(net.state_dict(), save_path)


if __name__ == '__main__':
    main()
