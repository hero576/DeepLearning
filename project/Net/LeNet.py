import torch
import torch.nn as nn
import torch.nn.functional as F

class Lenet5(nn.Module):
    def __init__(self):
        super(Lenet5,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,6,kernel_size=5,stride=1,padding=0),
            nn.AvgPool2d(kernel_size=2,stride=2,padding=0),
            nn.Conv2d(6,16,kernel_size=5,stride=1,padding=0),
            nn.AvgPool2d(kernel_size=2,stride=2,padding=0),
            nn.Flatten(),
            nn.Linear(16*5*5,120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10),
        )

    def forward(self,x):
        batch_size = x.size(0)
        x = self.model(x)
        return x

if __name__ == '__main__':
    net = Lenet5()
    tmp = torch.randn(2, 3, 32, 32)
    out = net(tmp)
    print(out.shape)
