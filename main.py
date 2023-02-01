import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class BestFitLine:
    df = pd.read_csv("smhi-data-mod.csv")
    x = None
    y = None

    @classmethod
    def create_matrix(cls):
        cls.x = np.array(BestFitLine.df.temp)
        cls.y = np.array(range(1, len(BestFitLine.df.month) + 1), dtype=float)
        return cls.x, cls.y


BestFitLine.create_matrix()

A = np.array([BestFitLine.x, np.ones(len(BestFitLine.x))], dtype=float).T
b = BestFitLine.y.reshape(-1, 1)

# Skapa normalekvationen
A_trans = A.T
A = np.dot(A_trans, A)
b = np.dot(A_trans, b)

Ab_norm = np.concatenate((A, b), axis=1)
rows = np.shape(Ab_norm)[0]
cols = np.shape(Ab_norm)[1]

x = np.zeros(cols - 1)

for i in range(cols - 1):
    for j in range(i + 1, rows):
        Ab_norm[j, :] = -(Ab_norm[j, i] / Ab_norm[i, i]) * Ab_norm[i, :] + Ab_norm[j, :]

for i in np.arange(rows - 1, -1, -1):
    x[i] = (Ab_norm[i, -1] - np.dot(Ab_norm[i, 0:cols - 1], x)) / Ab_norm[i, i]

k = np.round(x[0], 3)
m = np.round(x[1], 3)


def create_graph():
    plt.title("Medeltemperaturer 2022")
    plt.xlabel("Temperature")
    plt.ylabel("Month")
    plt.scatter(BestFitLine.x, BestFitLine.y, label="Medeltemperatur")
    plt.plot(BestFitLine.x, k * BestFitLine.x + m, "-r", label=f"Best line fit\nwhere;\nK = {k}\nm = {m}")
    plt.legend()
    return plt.show()


create_graph()
