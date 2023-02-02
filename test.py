import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Class for the raw data and the creation of matrices with that data
class RawData:
    df = None
    x = None
    y = None

    @classmethod
    def import_csv(cls, csv_file):
        cls.df = pd.read_csv(csv_file)

    @classmethod
    def create_matrix(cls, df_x, df_y):
        cls.x = np.array(df_x)

        for data_type in df_y:
            if isinstance(data_type, str):
                print(f"The data is of type 'string', filling array with floats for the len(df_y).")
                cls.y = np.array(range(1, len(df_y) + 1), dtype=float)

            elif isinstance(data_type, float) or isinstance(data_type, int):
                print(f"The data is of type: {type(data_type)}, no other actions taken.")
                cls.y = np.array(df_y, dtype=float)


class Calculations:
    rows = None
    cols = None
    ab_norm = None
    k = None
    m = None

    @classmethod
    # Tar in x och y från rå-data som parameter för att skapa matriser som vi kan göra beräkningar på
    def prepare_arrays(cls, x, y):
        a = np.array([x, np.ones(len(x))], dtype=float).T
        b = y.reshape(-1, 1)

        # Skapa normalekvationen
        a_trans = a.T
        a = np.dot(a_trans, a)
        b = np.dot(a_trans, b)

        cls.ab_norm = np.concatenate((a, b), axis=1)
        cls.rows = np.shape(cls.ab_norm)[0]
        cls.cols = np.shape(cls.ab_norm)[1]

    @classmethod
    def gauss_and_least_square(cls):
        solution_vector = np.zeros(cls.cols - 1)
        for i in range(cls.cols - 1):
            for j in range(i + 1, cls.rows):
                cls.ab_norm[j, :] = -(cls.ab_norm[j, i] / cls.ab_norm[i, i]) * cls.ab_norm[i, :] + cls.ab_norm[j, :]

        # Backwards substitution
        for i in np.arange(cls.rows - 1, -1, -1):
            solution_vector[i] = (cls.ab_norm[i, -1] - np.dot(cls.ab_norm[i, 0:cls.cols - 1], solution_vector)) \
                                 / cls.ab_norm[i, i]

        cls.k = np.round(solution_vector[0], 3)
        cls.m = np.round(solution_vector[1], 3)


def create_graph():
    plt.title("Medeltemperaturer 2022")
    plt.xlabel("Temperature")
    plt.ylabel("Month")
    plt.scatter(RawData.x, RawData.y, label="Medeltemperatur")
    plt.plot(RawData.x, Calculations.k * RawData.x + Calculations.m, "-r",
             label=f"Best line fit\nwhere;\nK = {Calculations.k}\nm = {Calculations.m}")
    plt.legend()
    return plt.show()


RawData.import_csv("smhi-data.csv")
RawData.create_matrix(RawData.df.temp, RawData.df.month)
Calculations.prepare_arrays(RawData.x, RawData.y)
Calculations.gauss_and_least_square()
create_graph()
