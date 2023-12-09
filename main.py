import numpy as np
from network import *
import os
current_dir = os.getcwd()
from tensorflow.keras.datasets import mnist

(x_train, Y_train), (x_test, y_test) = mnist.load_data(path = current_dir+'/mnist.npz')
x_train = x_train / 255
y_train = []

for i in Y_train:
  y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  y[i] = 1
  y_train.append(y)

y_train = np.array(y_train)
x_train_flat = x_train.reshape(60000, 784)

model = Network()
model.add(Dense(784, 32, ReLU))
model.add(Dense(32, 10, ReLU))

batch_size = 60

x_batch = x_train_flat.reshape(int(60000 / batch_size), batch_size, 784)
y_batch = y_train.reshape(int(60000 / batch_size), batch_size, 10)

population = model.create_population(200) # population size must be even

for data, labels in zip(x_batch, y_batch):
  population = model.train_reinforcement(data, labels, population)
