# 2018-10-15
# p.112 最適なパラメータを見つけよう
# グリッドサーチ
# gridSearch.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

# アヤメデータの読み込み
# ファイルDL
import urllib.request as req

url = "https://raw.githubusercontent.com/pandas-dev/pandas/master/pandas/tests/data/iris.csv"
savefile = "iris.csv"
req.urlretrieve(url, savefile)
iris_data = pd.read_csv(savefile, encoding="utf-8")

# アヤメデータをラベルと入力データに分離する
y = iris_data.loc[:, "Name"]
x = iris_data.loc[:, ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]

# 学習用とテスト用に分離する
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, train_size=0.8, shuffle=True
)

# グリッドサーチで利用するパラメータを指定 --- (*1)
parameters = [
    {"C": [1, 10, 100, 1000], "kernel": ["linear"]},
    {"C": [1, 10, 100, 1000], "kernel": ["rbf"], "gamma": [0.001, 0.0001]},
    {"C": [1, 10, 100, 1000], "kernel": ["sigmoid"], "gamma": [0.001, 0.0001]},
]

# グリッドサーチを行う --- (*2)
kfold_cv = KFold(n_splits=5, shuffle=True)
clf = GridSearchCV(SVC(), parameters, cv=kfold_cv)

clf.fit(x_train, y_train)
print("最適なパラメータ = ", clf.best_estimator_)

# 最適なパラメータで評価 --- (*3)
y_pred = clf.predict(x_test)
print("評価時の正解率 = ", accuracy_score(y_test, y_pred))

# メモ
# なんかあんまり精度上がってる感じしないな。
# 100％のときもあれば90%のときもある。
# グリッドサーチの他にランダムサーチというのもある。(RandomizedSearchCV)
# 自分で設定する必要のあるパラメータを「ハイパラメータ」という。
# グリッドサーチは適切なハイパラメータを探す手段の一つ
