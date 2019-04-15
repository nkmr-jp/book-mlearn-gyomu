# cut-resize.py
import matplotlib.pyplot as plt
import cv2

# 画像を読み込む
img = cv2.imread("test.jpg")
# 画像の一部を切り取る
im2 = img[150:450, 150:450]  # こんな書き方できるのか。配列っぽい。
# 画像をリサイズ
# im2 = cv2.resize(im2, (400, 400))
# リサイズした画像を保存
cv2.imwrite("cut-resize.png", im2)

# 画像を表示
# plt.axis("off")
plt.imshow(cv2.cvtColor(im2, cv2.COLOR_BGR2RGB))
plt.show()

# memo
# ※ OpenCVから読み出した画像はNumpy形式の配列データとなる、Pythonから手軽に操作できる。
# OpenCVはマルチプラットフォームの画像動画処理ライブラリ
# 機械学習では、いろくうかんの変換と切り取り、リサイズなどの操作をよく行う。
