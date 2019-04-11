# over30.py
import matplotlib.pyplot as plt
import pandas as pd

# ファイルを読む
df = pd.read_csv('kion10y.csv', encoding="utf-8")
# 気温が30度超えのデータを調べる ---(*1)
atui_bool = df["気温"] > 30
# データを抜き出す ---(*2)
atui = df[atui_bool]  # こんなかんじで条件指定して抽出できるんだ。便利
# 年ごとにカウント ---(*3)
cnt = atui.groupby(["年"])["年"].count()
# 出力
print(cnt)
cnt.plot()
plt.savefig("tenki-over30.png")
plt.show()

# メモ
# pandas便利だな
