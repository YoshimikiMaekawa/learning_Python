import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from sklearn.linear_model import LogisticRegression
import time

start_time = time.time()

# データの読み込み
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 小数化
x_train = x_train / 255
x_test = x_test / 255

# データ数
m_train, m_test = x_train.shape[0], x_test.shape[0]

# ベクトル化
x_train, x_test = x_train.reshape(m_train, -1), x_test.reshape(m_test, -1)

# ノルムで標準化
x_train = x_train / np.linalg.norm(x_train, ord=2, axis=1, keepdims=True)
x_test = x_test / np.linalg.norm(x_test, ord=2, axis=1, keepdims=True)

# fitting
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)

print("Elapsed[s]: ", time.time() - start_time)
print("Train: ", log_reg.score(x_train, y_train))
print("Test: ", log_reg.score(x_test, y_test))