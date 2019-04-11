# 2018-10-10
# count_wine_data.py
# p.86

import matplotlib.pyplot as plt
import pandas as pd

# ワインデータの読み込み
wine = pd.read_csv("winequality-white.csv", sep=";", encoding="utf-8")

# 品質データごとにグループ分けして、その数を数える
count_data = wine.groupby('quality')["quality"].count()  # データフレームはSQLっぽいことできるんだな。
print(count_data)

# 数えたデータをグラフに描画
count_data.plot()
plt.savefig("wine-count-plt.png")
plt.show()

# 2以下と１０は存在すらしていない。データ量にかなりばらつきがある。
# このように分布数に差のあるデータを 「不均衡データ」という。
# p.87


