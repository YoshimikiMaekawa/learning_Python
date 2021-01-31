import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# カラー画像の特性を保持したままできないものか...と作ったやつ
def trial_scaler(tensor):
    return (1000000 * tensor[:, :, :, 0] + 1000 * tensor[:, :, :, 1] + 1.0 * tensor[:, :, :, 2]) / 255255255

def decode_trial_scaler(tensor):
    decode_tensor = np.zeros((len(tensor), len(tensor[0]), len(tensor[0][0]), 3))
    for i in range(len(tensor)):
        for j in range(len(tensor[i])):
            for k in range(len(tensor[i][j])):
                color = "{:,d}".format(int(tensor[i][j][k] * 255255255 * 255)).split(",")
                red = int(color[0])
                green = int(color[1])
                blue = int(color[2])
                decode_tensor[i][j][k][0] = red
                decode_tensor[i][j][k][1] = green
                decode_tensor[i][j][k][2] = blue
    return decode_tensor / 255

# 画像の読み込み
image = np.array(Image.open("lenna.jpg"))
image = image / 255

# 4階のテンソルにする
image = image[np.newaxis, :, :, :]

# エセグレースケール化
trial_image = trial_scaler(image)
decode_image = decode_trial_scaler(trial_image)

# 表示
fig = plt.figure(figsize=(8, 8))
fig.subplots_adjust(hspace=0, wspace=0.2)
ax = fig.add_subplot(1, 3, 1)
ax.imshow(image[0])
ax = fig.add_subplot(1, 3, 2)
ax.imshow(trial_image[0], cmap="gray")
ax = fig.add_subplot(1, 3, 3)
ax.imshow(decode_image[0])
plt.show()