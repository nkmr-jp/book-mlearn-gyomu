# imshow.py

# ダウンロードした画像を画面に表示する
import matplotlib.pyplot as plt

# 画像のダウンロード
import urllib.request as req

url = "https://github.com/kujirahand/book-mlearn-gyomu/blob/master/src/ch3/cv2io/test.jpg?raw=true"
req.urlretrieve(url, "test.jpg")


import cv2

img = cv2.imread("test.jpg")
# print(img)

plt.imshow(img)  # 元の画像
plt.show()
plt.axis("off")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # BGRからRGBに変換してから表示している
plt.show()

# memo
# matplotlibのカラーデータは RGBの順で並んでいることを前提としている
