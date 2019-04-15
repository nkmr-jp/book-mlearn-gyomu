# %%
import urllib.request as req

req.urlretrieve(
    "https://github.com/kujirahand/book-mlearn-gyomu/raw/master/src/ch3/video/fish.mp4",
    "fish.mp4",
)

# %%

# 2018-10-22
# 動画から熱帯魚が映った場面を抽出しよう
# fishvideo_extract_diff.py
# p.172

import cv2, os

img_last = None  # 前回の画像
no = 0  # 画像の枚数
save_dir = "./exfish"  # 保存ディレクトリ名
if not os.path.exists(save_dir):
    os.mkdir(save_dir)  # ディレクトリを作成

# %%

# 動画ファイルから入力を開始 --- (*1)
cap = cv2.VideoCapture("fish.mp4")  # 0だとカメラからの入力になるが、動画ファイルを指定もできる
while True:
    # 画像を取得
    is_ok, frame = cap.read()
    if not is_ok:
        break
    frame = cv2.resize(frame, (640, 360))
    # 白黒画像に変換 --- (*2)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (15, 15), 0)
    img_b = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
    # 差分を確認する
    if not img_last is None:
        frame_diff = cv2.absdiff(img_last, img_b)  # --- (*3)
        # ※ OpenCV3の場合は[1]
        cnts = cv2.findContours(frame_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[
            0
        ]
        # 差分があった領域をファイルに出力 --- (*4)
        for pt in cnts:
            x, y, w, h = cv2.boundingRect(pt)
            if w < 100 or w > 500:
                continue  # ノイズを除去
            # 抽出した領域を画像として保存
            imgex = frame[y : y + h, x : x + w]
            outfile = save_dir + "/" + str(no) + ".jpg"
            cv2.imwrite(outfile, imgex)
            no += 1
    img_last = img_b
cap.release()
print("ok")

# めも
# とりあえず、差分のあったところ（動いたとこ)だけ抽出
# これだとさかな以外もたくさん抽出してしまう。
# p. 173

# 試せないけど、これの前のやつ。カメラから入力してリアルタイムで検出するやつのほうが面白そうだな。

