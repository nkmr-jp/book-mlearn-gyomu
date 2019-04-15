# %%
import urllib.request as req

req.urlretrieve(
    "https://raw.githubusercontent.com/kujirahand/book-mlearn-gyomu/master/src/ch3/cv2io/test.jpg",
    "test.jpg",
)

# %%
import matplotlib.pyplot as plt
import cv2

# p.182

# %%
# # 画像を読み込む
img = cv2.imread("test.jpg")

print(type(img))

# %%
# ネガポジ反転
img = 255 - img

# # 画像を表示
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

# %%
# 色空間をグレースケールに変換
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()


# %%
# p.183
# すべての色定数を表示
[i for i in dir(cv2) if i.startswith('COLOR_')]


# p.186 他にも回転とか反転とかいろいろできる。
