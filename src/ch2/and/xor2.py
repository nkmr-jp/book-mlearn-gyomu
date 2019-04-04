# %%

# 2018-10-09
# xor2.py
# p.66

# ライブラリのインポート --- (*1)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 学習用のデータと結果の準備
# X , Y
learn_data = [[0, 0], [1, 0], [0, 1], [1, 1]]
# X xor Y
learn_label = [0, 1, 1, 0]  # (*) xor用のラベルに変更

# アルゴリズムの指定(KNeighborsClassifier) --- (*2)
clf = KNeighborsClassifier(n_neighbors=1)

# 学習用データと結果の学習
clf.fit(learn_data, learn_label)

# テストデータによる予測
test_data = [[0, 0], [1, 0], [0, 1], [1, 1]]
test_label = clf.predict(test_data)

# テスト結果の評価
print(test_data, "の予測結果：", test_label)
print("正解率 = ", accuracy_score([0, 1, 1, 0], test_label))  # (*) xor用のラベルに変更


# メモ
# ここで用いたのはK近傍法(多クラス分類) KNN(K Nearest Neighbor)
# https://qiita.com/yshi12/items/26771139672d40a0be32

# 学習データをベクトル空間上にプロットしておき、未知のデータが得られたら、そこから距離が近い順に任意のK個を取得し、多数決でデータが属するクラスを推定する。
