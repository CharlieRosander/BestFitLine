import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from backend import RawData, Calculations, create_graph


if __name__ == "__main__":
    RawData.import_csv("smhi-data-int.csv")
    RawData.create_arrays(RawData.df.temp, RawData.df.month)
    Calculations.prepare_arrays(RawData.x, RawData.y)
    Calculations.gauss_and_least_square()
    create_graph(RawData.x, RawData.y, Calculations.k, Calculations.m)
