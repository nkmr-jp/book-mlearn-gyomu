# 2018-10-13
# クロスバリデーション

import pandas as pd
from sklearn.utils.testing import all_estimators
from sklearn.model_selection import KFold
import warnings
from sklearn.model_selection import cross_val_score

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

# classifierのアルゴリズム全てを取得する
warnings.filterwarnings('ignore')
allAlgorithms = all_estimators(type_filter="classifier")

# K分割クロスバリデーション用オブジェクト --- (*1)
kfold_cv = KFold(n_splits=5, shuffle=True)  # 5分割する

results = []
for (name, algorithm) in allAlgorithms:
    # 各アリゴリズムのオブジェクトを作成
    clf = algorithm()

    # scoreメソッドをもつクラスを対象とする--- (*2)
    if hasattr(clf, "score"):  #  scoreのメソッドを持つクラスのみに限定

        # クロスバリデーションを行う--- (*3)
        # 5回評価して、それぞれの正解率を配列で返す
        scores = cross_val_score(clf, x, y, cv=kfold_cv)  # cvは整数でも指定できる
        # print(name, "の正解率=")
        # print(scores)
        results.append({"アルゴリズム": name, "正解率": scores.mean()})  # スコアの平均値を出す


# %%
df = pd.DataFrame(results)
df.sort_values('正解率', ascending=False)

# メモ
# K分割クロスバリデーション
# cross_val_scoreで使える。
# データをK個に分割してK回学習評価する。
# K-1個を学習用データ。 1個を評価用データにする。
# 5グループに分割したら4個が学習用データ、1つが評価用データ。
# 入れ替えて一巡させる。
# p.110-112

# LinearDiscriminantAnalysis, SVC, QuadraticDiscriminantAnalysis, MLPClassifier あたりが有力
