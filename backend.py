import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Class for the raw data and the creation of arrays with that data
class RawData:
    df = None
    x = None
    y = None

    # Importerar csv filen som en DataFrame
    @classmethod
    def import_csv(cls, csv_file):
        cls.df = pd.read_csv(csv_file)

    # Funktion för att skapa arraysen med värdena från DataFrame:en
    @classmethod
    def create_arrays(cls, df_x, df_y):
        cls.x = np.array(df_x)

        # En for-loop med if-satser som kollar vilken datatyp det är i kolumnen "month"
        # Om typen är string så gör den en array på längden av värdena i kolumnen som float
        # Annars tar den in dom faktiska värdena och konverterar till float
        for data_type in df_y:
            if isinstance(data_type, str):
                cls.y = np.array(range(1, len(df_y) + 1), dtype=float)

                return print(f"The data imported is of type 'string', filling array with floats for range(len(df_y)).")
            elif isinstance(data_type, float) or isinstance(data_type, int):
                cls.y = np.array(df_y, dtype=float)

                return print(f"The data imported is of type: {type(data_type)}, converted values to dtype float.")


# En klass för beräkningarna
class Calculations:
    rows = None
    cols = None
    ab_norm = None
    k = None
    m = None

    @classmethod
    # Tar in x och y från rå-data som parameter för att skapa matriser
    def prepare_arrays(cls, x, y):
        a = np.array([x, np.ones(len(x))], dtype=float).T
        b = y.reshape(-1, 1)

        # Skapa normalekvationen genom a-transponat * a och a-transponat * b och slår sedan ihop dom till en matris
        a_trans = a.T
        a = np.dot(a_trans, a)
        b = np.dot(a_trans, b)

        cls.ab_norm = np.concatenate((a, b), axis=1)
        cls.rows = np.shape(cls.ab_norm)[0]
        cls.cols = np.shape(cls.ab_norm)[1]

    # Funktion för att utföra gauss och minstakvadratmetoden på matrisen.
    # Först körs gausselimination för att nolla upp ifrån och ner, vänster till höger
    # och sedan körs "backwards substitution" för att göra samma sak fast tvärtom
    @classmethod
    def gauss_and_least_square(cls):

        # En variabel för att spara resultaten av uträkningarna
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


# Plottar ut grafen
def create_graph(x_raw, y_raw, k, m):
    plt.title("Medeltemperaturer 2022")
    plt.xlabel("Temperature")
    plt.ylabel("Month")
    plt.scatter(x_raw, y_raw, label="Medeltemperatur")
    plt.plot(x_raw, k * x_raw + m, "-r",
             label=f"Best line fit\nwhere;\nK = {k}\nm = {m}")
    plt.legend()
    return plt.show()
