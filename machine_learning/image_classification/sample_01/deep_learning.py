from keras.models import Sequential
from keras.layers import Dense
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

# ベクトル化
x_train, x_test = x_train.reshape(m_train, -1), x_test.reshape(m_test, -1)

# L2ノルムで標準化
x_train = x_train / np.linalg.norm(x_train, ord=2, axis=1, keepdims=True)
x_test = x_test / np.linalg.norm(x_test, ord=2, axis=1, keepdims=True)

# yをone-hotベクトル化
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

# モデルの定義
print(x_train.shape)
model = Sequential()
model.add(Dense(768, activation="relu", input_shape=x_train.shape[1:]))
model.add(Dense(192, activation="relu"))
model.add(Dense(10, activation="softmax"))

# compile
model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

# fitting
history = model.fit(x_train, y_train, batch_size=64, epochs=15).history

# elapsed time
print("Elapsed[s] : ", time.time() - start_time)

# accuracy
test_eval = model.evaluate(x_test, y_test)
print("train accuracy :", history["accuracy"][-1])
print("test accuracy :", test_eval[1])

# plot
plt.plot(range(len(history["loss"])), history["loss"], marker=".")
plt.show()