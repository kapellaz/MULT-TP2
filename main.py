"""
Bruno Sequeira: 2020235721
Rui Santos: 2020225542
Tomás Dias: 2020215701
"""

import numpy as np
import librosa as lb


# exercicio 1

def read_csv():

    data = np.genfromtxt(
        'Features - Audio MER/top100_features.csv', dtype=np.str_, delimiter=",")

    linhas, colunas = np.shape(data)

    fNames = data[0, 1:colunas - 1]

    data = data[1::, 1:(colunas - 1)].astype(float)

    dataNormalizada = normaliza(data)

    np.savetxt("Features - Audio MER/top100_features_normalized.csv",
               dataNormalizada, fmt="%lf", delimiter=",")


def normaliza(data):
    fn = np.zeros(data.shape)
    for i in range(len(data[0])):
        max = np.max(data[::, i])
        min = np.min(data[::, i])
        if (max == min):
            fn[::, i] = 0
        else:
            fn[::, i] = (data[::, i] - min)/(max - min)

    return fn


def main():
    read_csv()


if __name__ == "__main__":

    main()
