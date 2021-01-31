import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10

# データの読み込み
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 小数化
x_train = x_train / 255
x_test = x_test / 255

# 画像の表示(10クラス分を10枚ずつ表示)
fig = plt.figure(figsize = (8, 8))
fig.subplots_adjust(hspace = 0, wspace = 0)
for i in range(10):
    index, count = 0, 0
    while(True):
        if y_train[index] == i:
            ax = fig.add_subplot(10, 10, i*10+count+1, xticks = [], yticks = [])
            ax.imshow(x_train[index])
            count += 1
        if 10 <= count: break
        index +=1
plt.show()