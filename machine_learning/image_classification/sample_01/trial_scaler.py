import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import colorsys

# カラー画像の特性を保持したままできないものか...と作ったやつ
def trial_scaler(tensor):
    return (1000000 * tensor[:, :, :, 0] + 1000 * tensor[:, :, :, 1] + 1.0 * tensor[:, :, :, 2]) / 255255255

def decode_trial_scaler(tensor):
    decode_tensor = np.zeros((len(tensor), len(tensor[0]), len(tensor[0][0]), 3))
    for i in range(len(tensor)):
        for j in range(len(tensor[i])):
            for k in range(len(tensor[i][j])):
                color = int(tensor[i][j][k] * 255255255 * 255)
                decode_tensor[i][j][k][0] = int(color / 1000000)
                decode_tensor[i][j][k][1] = int((color % 1000000) / 1000)
                decode_tensor[i][j][k][2] = int((color % 1000000) % 1000)
    return decode_tensor / 255

def HSVColor(img):
    if isinstance(img, Image.Image):
        r, g, b = img.split()
        Hdat = []
        Sdat = []
        Vdat = [] 
        for rd,gn,bl in zip(r.getdata(), g.getdata(), b.getdata()) :
            h, s, v = colorsys.rgb_to_hsv(rd/255.,gn/255.,bl/255.)
            Hdat.append(int(h*255.))
            Sdat.append(int(s*255.))
            Vdat.append(int(v*255.))
        r.putdata(Hdat)
        g.putdata(Sdat)
        b.putdata(Vdat)
        return Image.merge('RGB', (r, g, b))
    else:
        return None

# 画像の読み込み
# image = np.array(Image.open("lenna.jpg"))
image = np.array(HSVColor(Image.open("lenna.jpg")))
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