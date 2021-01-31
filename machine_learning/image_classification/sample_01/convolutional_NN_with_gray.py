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
(x_train_origin, y_train), (x_test_origin, y_test) = cifar10.load_data()

# 小数化
x_train_origin = x_train_origin / 255
x_test_origin = x_test_origin / 255

# データ数
m_train, m_test = x_train_origin.shape[0], x_test_origin.shape[0]

# グレースケール化
def gray_scaler(tensor):
    return 0.2126 * tensor[:, :, :, 0] + 0.7152 * tensor[:, :, :, 1] + 0.0722 * tensor[:, :, :, 2]

# エセグレースケール化
def trial_scaler(tensor):
    return (1000000 * tensor[:, :, :, 0] + 1000 * tensor[:, :, :, 1] + 1.0 * tensor[:, :, :, 2]) / 255255255

# x_train, x_test = gray_scaler(x_train_origin), gray_scaler(x_test_origin)
x_train, x_test = trial_scaler(x_train_origin), trial_scaler(x_test_origin)

x_train = x_train.reshape(m_train, 32, 32, 1)
x_test = x_test.reshape(m_test, 32, 32, 1)

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