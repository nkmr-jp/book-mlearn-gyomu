# 2018-10-11
# p.93
# 過去の気象データDL
# https://www.data.jma.go.jp/gmd/risk/obsdl/index.php
# csv_trim_header.py

dir = "src/ch2/4_tenki/"
in_file = dir + "data.csv"
out_file = dir + "kion10y.csv"

# CSVファイルを一行ずつ読み込み ---(*1)
with open(in_file, "rt", encoding="Shift_JIS") as fr:
    lines = fr.readlines()

# ヘッダをそぎ落として、新たなヘッダをつける ---(*2)
lines = ["年,月,日,気温,品質,均質\n"] + lines[5:]
# lines
lines = map(lambda v: v.replace('/', ','), lines)
result = "".join(lines).strip()  # 文字列として連結
print(result)
# result

# # 結果をファイルへ出力 ---(*3)
with open(out_file, "wt", encoding="utf-8") as fw:
    fw.write(result)
    print("saved.")

# メモ
# 日本のデータは前処理めんどいな。文字コード変換と改行コード変換とか

# %%
# URLからダウンロードする方法
import pandas as pd
from urllib.request import urlretrieve

urlretrieve(
    "https://raw.githubusercontent.com/kujirahand/mlearn-sample/master/tenki2006-2016/kion10y.csv",
    "kion10y.csv",
)

pd.read_csv("kion10y.csv")

# メモ
# 公開されてるデータを手動でDLするのだるいから、基本このやり方かな。
