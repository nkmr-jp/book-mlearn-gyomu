# 2018-10-19

# p.142
import matplotlib.pyplot as plt

# 手書きのデータを書き込む
from sklearn import datasets

digits = datasets.load_digits()

# 15個連続で出力する
for i in range(15):
    plt.subplot(3, 5, i + 1)
    plt.axis("off")
    plt.title(str(digits.target[i]))
    plt.imshow(digits.images[i], cmap="gray")

plt.show()

# %%
# 画像を機械学習しよう
# ml_digits.py
# p.145

from sklearn.model_selection import train_test_split
from sklearn import datasets, svm, metrics
from sklearn.metrics import accuracy_score

# データを読み込む --- (*1)
digits = datasets.load_digits()
x = digits.images
y = digits.target
x = x.reshape((-1, 64))  # 二次元配列を一次元配列に変換 --- (*2)

# データを学習用とテスト用に分割する --- (*3)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# データを学習 --- (*4)
clf = svm.LinearSVC()
clf.fit(x_train, y_train)

# 予測して精度を確認する --- (*5)
y_pred = clf.predict(x_test)
print(accuracy_score(y_test, y_pred))

# %%
# memo
# 09:28:43
# x.reshape((-1, 64)) # これの数値の意味調べる (-1,64)の次元の配列に変換

# 2018-10-20

# 学習済みデータを保存しよう
# p.146
from sklearn.externals import joblib

joblib.dump(clf, 'digits.pkl')
# 学習済みデータを読み込み
clf = joblib.load('digits.pkl')
clf
