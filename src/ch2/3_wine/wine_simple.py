# 2018-10-10
# wine_sample.py
# p.84

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # ランダムフォレストのアルゴリズムで学習する
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# データを読み込む
wine = pd.read_csv("winequality-white.csv", sep=";", encoding="utf-8")

# データをラベルとデータに分離 ---(*1)
y = wine["quality"]  # ラベル
x = wine.drop("quality", axis=1)  # クオリティの列のみ削除

# 学習用とテスト用に分割する ---(*2)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2
)  # テストサイズだけでいいんだ

# 学習する ---(*3)
model = RandomForestClassifier()
model.fit(x_train, y_train)

# 評価する ---(*4)
y_pred = model.predict(x_test)  #  予測した ラベル
print(classification_report(y_test, y_pred))  # 各分類ラベルごとのレポート
# precision:精度    recall:再現率  f1-score:精度と再現率の調和平均   support:正解ラベルのデータ数
# このあたりAIdemiyでやったな。

print("正解率=", accuracy_score(y_test, y_pred))  # テスト用のラベルを使って予測を評価

# x = データ
# y = ラベル
# x（データ）から、y（ラベル）を予測する

# p.86 2018-10-10
