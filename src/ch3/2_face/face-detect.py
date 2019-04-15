# %%
# 正面の顔を検出するカスケードファイル
# 画像のダウンロード
import urllib.request as req

req.urlretrieve(
    "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt.xml",
    "haarcascade_frontalface_alt.xml",
)
req.urlretrieve(
    "https://raw.githubusercontent.com/kujirahand/book-mlearn-gyomu/master/src/ch3/face/girl.jpg",
    "girl.jpg",
)

# %%
# 2018-10-17
# p.132
# 最新の顔検出カスケードファイルをダウンロードする
# https://github.com/opencv/opencv/tree/master/data/haarcascades
# face-detect.py

import matplotlib.pyplot as plt
import cv2

# カスケードファイルを指定して検出器を作成 --- (*1)
cascade_file = "haarcascade_frontalface_alt.xml"
cascade = cv2.CascadeClassifier(cascade_file)

# 画像の読み込んでグレイスケールに変換する --- (*2)
img = cv2.imread("girl.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 顔認識を実行 --- (*3)
face_list = cascade.detectMultiScale(img_gray, minSize=(150, 150))  # 顔と認識する領域の最小サイズ
# 結果を確認 --- (*4)
if len(face_list) == 0:
    print("失敗")
    quit()
# 認識した部分に印をつける --- (*5)
for (x, y, w, h) in face_list:
    print("顔の座標=", x, y, w, h)
    red = (0, 0, 255)
    cv2.rectangle(img, (x, y), (x + w, y + h), red, thickness=20)

# 画像を出力
cv2.imwrite("face-detect.png", img)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()


# メモ
# この辺はライブラリの使い方
# cv2にもClassifilerがある。
# p.134
# CascadeClassifier これは第一引数にcascade_fileを指定することで様々な物体検出ができる。
# すごい。
# グレースケールに変換するのは、物体の明暗で検知するため
