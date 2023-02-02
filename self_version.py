import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class RawData:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def create_matrix(self):
        df_x = self.df.temp
        df_y = self.df.month
        x = np.array(df_x)

        for data_type in df_y:
            if isinstance(data_type, str):
                y = np.array(range(1, len(df_y) + 1), dtype=float)
                print(f"The data is of type 'string', filling array with floats for range(len(df_y)).")
            elif isinstance(data_type, float) or isinstance(data_type, int):
                y = np.array(df_y, dtype=float)
                print(f"The data is of type: {type(data_type)}, converted values to dtype float.")

        return x, y


class Calculations:
    @staticmethod
    def prepare_arrays(x, y):
        a = np.array([x, np.ones(len(x))], dtype=float).T
        b = y.reshape(-1, 1)

        a_trans = a.T
        a = np.dot(a_trans, a)
        b = np.dot(a_trans, b)

        ab_norm = np.concatenate((a, b), axis=1)
        rows = np.shape(ab_norm)[0]
        cols = np.shape(ab_norm)[1]

        return ab_norm, rows, cols

    @staticmethod
    def gauss_and_least_square(ab_norm, rows, cols):
        solution_vector = np.zeros(cols - 1)
        for i in range(cols - 1):
            for j in range(i + 1, rows):
                ab_norm[j, :] = -(ab_norm[j, i] / ab_norm[i, i]) * ab_norm[i, :] + ab_norm[j, :]

        for i in np.arange(rows - 1, -1, -1):
            solution_vector[i] = (ab_norm[i, -1] - np.dot(ab_norm[i, 0:cols - 1], solution_vector)) \
                                 / ab_norm[i, i]

        k = np.round(solution_vector[0], 3)
        m = np.round(solution_vector[1], 3)

        return k, m


def create_graph(x, y, k, m):
    plt.title("Medeltemperaturer 2022")
    plt.xlabel("Temperature")
    plt.ylabel("Month")
    plt.scatter(x, y, label="Medeltemperatur")
    plt.plot(x, k * x + m, "-r")
    plt.show()

RawData.__init__("smhi-data.csv")
