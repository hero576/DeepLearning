
from keras.models import Sequential

from keras.layers import Dense,Activation

import numpy as np

import matplotlib.pylab as plt


class XOR:
    def __init__(self):
        self.model = self.generate_model()


    def generate_train_data(self):
        x_train = np.array([[1,1],[0,0],[0,1],[1,0]])
        y_train = np.array([[0],[0],[1],[1]])
        return x_train,y_train


    def generate_test_data(self):
        return self.generate_train_data()

    def generate_model(self):
        model = Sequential()
        model.add(Dense(units=2,input_dim=2))
        model.add(Activation("relu"))
        model.add(Dense(units=1))
        model.add(Activation("sigmoid"))

        model.compile(loss="binary_crossentropy",optimizer="sgd",metrics=["accuracy"])
        return model

    def train(self):
        x,y = self.generate_train_data()
        hist = self.model.fit(x,y,epochs=1000)
        plt.scatter(range(len(hist.history['loss'])),hist.history['loss'])
        plt.show()

    def test(self):
        x,y = self.generate_test_data()
        loss_metrics = self.model.evaluate(x,y)
        print(loss_metrics)

if __name__ == '__main__':
    x = XOR()
    x.train()
    x.test()


