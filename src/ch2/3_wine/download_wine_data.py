# %%
# 2018-10-10
# download_wine_data.py
# p.80
#
# UCI Machine Learning Repository!
# https://archive.ics.uci.edu/ml/index.php


from urllib.request import urlretrieve

url = (
    "https://archive.ics.uci.edu"
    + "/ml/machine-learning-databases/wine-quality"
    + "/winequality-white.csv"
)
savepath = "winequality-white.csv"
urlretrieve(url, savepath)

# %%
import pandas as pd

df = pd.read_csv('winequality-white.csv', sep=';')
# df
df = df.sort_values(by='quality', ascending=False)  # クオリティ順に並べ替え
# df[-1:]  # 最後の1つのみ取得
df[:10]
