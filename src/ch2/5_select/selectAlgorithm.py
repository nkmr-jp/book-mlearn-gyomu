# 2018-10-13
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
from sklearn.utils.testing import all_estimators

import warnings

warnings.filterwarnings('ignore')

# アヤメデータの読み込み
# from urllib.request import urlretrieve
# urlretrieve(
#     "https://raw.githubusercontent.com/kujirahand/book-mlearn-gyomu/master/src/ch2/select/iris.csv", "iris.csv"
# )

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
warnings.filterwarnings('ignore')
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.5, train_size=0.5, shuffle=True
)

# classifierのアルゴリズム全てを取得する --- (*1)
warnings.filterwarnings('ignore')
allAlgorithms = all_estimators(type_filter="classifier")

results = []
for (name, algorithm) in allAlgorithms:
    # 各アリゴリズムのオブジェクトを作成 --- (*2)
    clf = algorithm()

    # 学習して、評価する --- (*3)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    # print(name, "の正解率 = ", score)
    results.append({"アルゴリズム": name, "正解率": score})


# %%
df = pd.DataFrame(results)
df.sort_values('正解率', ascending=False)

# memo
# pandas超便利だな
# アルゴリズムもfit とpredictでインターフェースが統一されているからルーブで実行できる。
# 便利だ。ダックタイピング的
# all_estimators便利
