from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D


class LeNet(Model):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = Conv2D(16, 5, activation='relu')
        self.pool1 = MaxPool2D()
        self.conv2 = Conv2D(32, 5, activation='relu')
        self.pool2 = MaxPool2D()
        self.flatten = Flatten()
        self.d1 = Dense(120, activation='relu')
        self.d2 = Dense(84, activation='relu')
        self.d3 = Dense(10, activation='softmax')

    def call(self, x, **kwargs):
        x = self.pool1(self.conv1(x))       # input[batch, 32, 32, 3] output[batch, 14, 14, 16]
        x = self.pool2(self.conv2(x))       # input[batch, 14, 14, 16] output[batch, 5, 5, 32]
        x = self.flatten(x)                 # output [batch, 5*5*32]
        x = self.d1(x)                      # output [batch, 120]
        x = self.d2(x)                      # output [batch, 84]
        return self.d3(x)                   # output [batch, 10]
