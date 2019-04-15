# 2018-10-15
# OpenCV
# https://opencv.org/

# 2018-10-15
# download_imread.py
# p.122

# pip install opencv-python

# 画像のダウンロード
import urllib.request as req

url = "http://uta.pw/shodou/img/28/214.png"
req.urlretrieve(url, "test.png")

# OpenCVで読み込む
import cv2

img = cv2.imread("test.png")  # ※読み込み失敗してもNoneを返す。例外発生しないので注意
print(img)


# memo:
# OpenCVの機械学習との関わり
# OpenCVを使って画像形式や色数などを整える。 機械学習に与える入力では、画像サイズは同じサイズである必要がある。 リサイズしたり、切り出したりする必要がある。
#
# IoTでも
# 小規模な機械学習であればRasberry Pi 上でも動かすことができる。
