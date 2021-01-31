import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from sklearn.svm import LinearSVC
import time

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

# ベクトル化
x_train, x_test = x_train.reshape(m_train, -1), x_test.reshape(m_test, -1)

# L2ノルムで標準化
x_train = x_train / np.linalg.norm(x_train, ord=2, axis=1, keepdims=True)
x_test = x_test / np.linalg.norm(x_test, ord=2, axis=1, keepdims=True)

# サポートベクトルマシンで学習
svc = LinearSVC()
svc.fit(x_train, y_train)
print("Elapsed[s] : ", time.time() - start_time)
print("Train :", svc.score(x_train, y_train))
print("Test :", svc.score(x_test, y_test))