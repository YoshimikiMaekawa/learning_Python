from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Activation, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras.datasets import cifar10
from keras.utils.np_utils import to_categorical
import numpy as np
import time
import matplotlib.pyplot as plt

start_time = time.time()

# データの読み込み
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 小数化
x_train = x_train / 255
x_test = x_test /255

# データ数
m_train, m_test = x_train.shape[0], x_test.shape[0]

# yをone-hotベクトル化
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

# モデルの定義
print(x_train.shape)
model = Sequential()

# CONV -> RELU -> MAXPOOL
model.add(Conv2D(10, (3, 3), strides=(1, 1), input_shape=x_train.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPool2D(3, 3))

# CONV -> RELU -> BN -> Flatten
model.add(Conv2D(20, (3, 3), strides=(1, 1)))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=3))
model.add(Flatten())

model.add(Dense(10, activation="softmax"))

# compile
model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

# fitting
history = model.fit(x_train, y_train, batch_size=64, epochs=30).history

# elapsed time
print("Elapsed[s] : ", time.time() - start_time)

# accuracy
test_eval = model.evaluate(x_test, y_test)
print("train accuracy :", history["accuracy"][-1])
print("test accuracy :", test_eval[1])

# plot
plt.plot(range(len(history["loss"])), history["loss"], marker=".")
plt.show()