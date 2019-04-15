# 2018-10-23
# 機械学習で動画に熱帯魚が映っているベストな場面を見つけよう
# 手動で画像分類をする。まあこれも練習だな。やろう。

# 2018-10-2409:30:33
# とりあえず魚の画像は抽出した。
# 次は魚じゃない画像


# p.175
# fish_train.py

import cv2
import os, glob
from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

# 画像の学習サイズやパスを指定
image_size = (64, 32)
# path = os.path.dirname(os.path.abspath(__file__))

path_fish = 'fish_images/fish'
path_nofish = 'fish_images/nofish'
x = []  # 画像データ
y = []  # ラベルデータ

# 画像データを読み込んで配列に追加 --- (*1)
def read_dir(path, label):
    files = glob.glob(path + "/*.jpg")
    for f in files:
        img = cv2.imread(f)
        img = cv2.resize(img, image_size)
        img_data = img.reshape(-1)  # 一次元に展開
        x.append(img_data)
        y.append(label)


# 画像データを読み込む
read_dir(path_nofish, 0)
read_dir(path_fish, 1)

# データを学習用とテスト用に分割する --- (*2)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# データを学習 --- (*3)
clf = RandomForestClassifier()
clf.fit(x_train, y_train)

# 精度の確認 --- (*4)
y_pred = clf.predict(x_test)
print(accuracy_score(y_test, y_pred))

# データを保存 --- (*5)
joblib.dump(clf, 'fish.pkl')


# メモ
# 手動分類結構たいへんだったな
# 前処理が一番時間かかるか。やっぱ。

# 2019-04-15
# ここは面倒なので飛ばす。
