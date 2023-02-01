import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class BestFitLine:
    df = pd.read_csv("smhi-data-mod.csv")
    x = None
    y = None

    @classmethod
    def create_matrix(cls):
        cls.x = np.array(BestFitLine.df.temp).reshape(-1, 1)
        cls.y = np.array(range(1, len(BestFitLine.df.month) + 1), dtype=float)
        return cls.x, cls.y


BestFitLine.create_matrix()
print(BestFitLine.x)
# print(BestFitLine.x)
# Test med att reshape redan i x (ställa upp x),
# sedan använda concatenate och reshape på ones (ställa upp ones) och slå ihop dom på axis=1
# Bevarar datan omodifierad jämfört med transpose?
A = np.concatenate([BestFitLine.x, np.ones([len(BestFitLine.x)]).reshape(-1, 1)], axis=1, dtype=float)
b = BestFitLine.y.reshape(-1, 1)
print(A)
#
# # Skapa normalekvationen
# A_trans = A.T
# A = np.dot(A_trans, A)
# b = np.dot(A_trans, b)
#
# Ab_norm = np.concatenate((A, b), axis=1)
#
#
# # print(BestFitLine.x)
# # print(BestFitLine.y)
# print(A)
# print(b)
# print(Ab_norm)


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
#
# class BestFitLine:
#     df = pd.read_csv("smhi-data-mod.csv")
#     x = None
#     y = None
#
#     @classmethod
#     def create_matrix(cls):
#         cls.x = np.array(BestFitLine.df.temp)
#         cls.y = np.array(range(1, len(BestFitLine.df.month) + 1), dtype=float)
#         return cls.x, cls.y
#
#
# BestFitLine.create_matrix()
# print(BestFitLine.x)
# A = np.array([BestFitLine.x, np.ones(len(BestFitLine.x))], dtype=float).T
# b = BestFitLine.y.reshape(-1, 1)
# print(A)
#
# # Skapa normalekvationen
# A_trans = A.T
# A = np.dot(A_trans, A)
# b = np.dot(A_trans, b)
#
# Ab_norm = np.concatenate((A, b), axis=1)
# print(Ab_norm)
# rows = np.shape(Ab_norm)[0]
# cols = np.shape(Ab_norm)[1]
#
# x = np.zeros(cols-1)
# print(x)
# # print(x)
# # print(Ab_norm)
# for i in range(cols-1):
#     # print(i)
#     for j in range(i + 1, rows):
#         # print(j)
#         Ab_norm[j, :] = -(Ab_norm[j, i] / Ab_norm[i, i]) * Ab_norm[i, :] + Ab_norm[j, :]
#
# # print(Ab_norm)
#
# for i in np.arange(rows-1, -1, -1):
#     x[i] = (Ab_norm[i, -1] - Ab_norm[i, 0:cols - 1] @ x) / Ab_norm[i, i]
#
# k = np.round(x[0], 3)
# m = np.round(x[1], 3)
#
# print(x)
# print(k)
# print(m)
#
# plt.title("Medeltemperaturer 2022")
# plt.xlabel("Temperature")
# plt.ylabel("Month")
# plt.scatter(BestFitLine.x, BestFitLine.y, label="Medeltemperatur")
# plt.plot(BestFitLine.x, k * BestFitLine.x + m, "-r", label=f"Best linear fit\nwhere;\nK = {k}\nm = {m}")
# plt.legend()
# plt.show()
