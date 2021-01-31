import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# グレースケール化
def gray_scaler(tensor):
    return 0.2126 * tensor[:, :, :, 0] + 0.7152 * tensor[:, :, :, 1] + 0.0722 * tensor[:, :, :, 2]

# 画像の読み込み
image = np.array(Image.open("lenna.jpg"))
image = image / 255

# 4階のテンソルにする
image = image[np.newaxis, :, :, :]
print(image.shape)

# グレースケール化
gray_image = gray_scaler(image)

# 表示
fig = plt.figure(figsize=(8, 8))
fig.subplots_adjust(hspace=0, wspace=0.2)
ax = fig.add_subplot(1, 2, 1)
ax.imshow(image[0])
ax = fig.add_subplot(1, 2, 2)
ax.imshow(gray_image[0], cmap="gray")
plt.show()