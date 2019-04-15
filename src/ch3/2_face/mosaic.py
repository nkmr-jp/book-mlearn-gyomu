# 2018-10-18
# モザイクをかける
# mosaic.py
# p.135

import cv2


def mosaic(img, rect, size):
    """
    img : numpy.ndarray
    rect: tuple
    size: int
    """

    # モザイクをかける領域を取得
    (x1, y1, x2, y2) = rect
    w = x2 - x1
    h = y2 - y1
    i_rect = img[y1:y2, x1:x2]
    # 一度縮小して拡大する
    i_small = cv2.resize(i_rect, (size, size))
    i_mos = cv2.resize(i_small, (w, h), interpolation=cv2.INTER_AREA)  # エリアを指定してもとにもどす。
    # 画像にモザイク画像を重ねる
    img2 = img.copy()
    img2[y1:y2, x1:x2] = i_mos  # 指定したエリアに重ねる。
    return img2


# %%
# 画像のダウンロード
import urllib.request as req

req.urlretrieve(
    "https://raw.githubusercontent.com/kujirahand/book-mlearn-gyomu/master/src/ch3/face/cat.jpg",
    "cat.jpg",
)

# %%
import matplotlib.pyplot as plt
import cv2

# mosaic-test.py
# from mosaic import mosaic as mosaic

# 画像を読み込んでモザイクをかける
img = cv2.imread("cat.jpg")
mos = mosaic(img, (50, 50, 450, 450), 10)

# モザイクをかけた画像を出力
cv2.imwrite("cat-mosaic.png", mos)
plt.imshow(cv2.cvtColor(mos, cv2.COLOR_BGR2RGB))
plt.show()

# めも
# img = cv2.imread("cat.jpg") これはnumpyの配列になる
# types(img)
# numpy.ndarray


# %%
# 画像のダウンロード
import urllib.request as req

req.urlretrieve(
    "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt.xml",
    "haarcascade_frontalface_alt.xml",
)
req.urlretrieve(
    "https://raw.githubusercontent.com/kujirahand/book-mlearn-gyomu/master/src/ch3/face/family.jpg",
    "family.jpg",
)


# %%
# 2018-10-18 Next

# 人間の顔に自動でモザイクをかけよう
# p.136

import matplotlib.pyplot as plt
import cv2

# from mosaic import mosaic as mosaic

# カスケードファイルを指定して分類機を作成 --- (*1)
cascade_file = "haarcascade_frontalface_alt.xml"
cascade = cv2.CascadeClassifier(cascade_file)

# 画像の読み込んでグレイスケールに変換 --- (*2)
img = cv2.imread("family.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 顔検出を実行 --- (*3)
face_list = cascade.detectMultiScale(img_gray, minSize=(150, 150))
if len(face_list) == 0:
    quit()

# 認識した部分の画像にモザイクをかける --- (*4)
for (x, y, w, h) in face_list:
    img = mosaic(img, (x, y, x + w, y + h), 10)

# for (x,y,w,h) in face_list:
#     print("顔の座標=", x, y, w, h)
#     red = (0, 0, 255)
#     cv2.rectangle(img, (x, y), (x+w, y+h), red, thickness=20)

# 画像を出力
cv2.imwrite("family-mosaic.png", img)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

# メモ
# 横向きの顔は認識できない
