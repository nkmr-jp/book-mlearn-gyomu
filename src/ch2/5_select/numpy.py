# p.115
# numpy
import numpy as np

a = np.array([1, 2, 3, 4, 5])
print(a)
print(type(a))
b = np.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
print(b)

# %%
# 配列を0で初期化する
print(np.zeros(10))
print(np.zeros((3, 2)))
print(np.ones(10))

# %%
# 連番の配列を作る
print(np.arange(5))
print(np.arange(2, 9))
print(np.arange(2, 9, 0.5))

# %%
# p.116
import numpy as np

# 行列計算
a = np.array([1, 2, 3, 4, 5])
print(a * 2)  # 全要素2倍する

x = np.arange(10)
y = 3 * x + 5
print(y)


# %%
# 次元数を調べる
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a.shape)

b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(b.shape)


# %%
# 次元数を変換する
print('a=', a)
print('b=', a.flatten())


# %%
print(a)
print(a.reshape(3, 2))  # これすごいな
print(a.reshape(6, 1))
print(a.reshape(1, 6))
print(a.reshape(2, 3))


# %%
v = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
a = v[0]
b = v[1:]
c = v[:, 0]  # 縦は全部。横は1列目のみ
print("a=", a)
print("b=", b)
print("c=", c)

# 便利だな
