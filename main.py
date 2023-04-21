"""
Bruno Sequeira: 2020235721
Rui Santos: 2020225542
Tomás Dias: 2020215701
"""

import numpy as np
import librosa as lb
import os
import scipy.stats as scp
import warnings
import sys

SR = 22050

Calcula3_1 = True
Features900 = "MER_audio_taffc_dataset\FeaturesQuadrantes"


# Exercicio 1
def stats(array):
    mean = np.mean(array)
    std = np.std(array)
    skewness = scp.skew(array)
    kurtosis = scp.kurtosis(array)
    media = np.median(array)
    max = np.max(array)
    min = np.min(array)

    return np.array([mean, std, skewness, kurtosis, media, max, min])


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


# Exercicio 2
def extractLibrosa2_1_1():
    #features = np.zeros(900,190)

    matrix = np.zeros((900, 190), dtype=np.float64)

    l = -1
    for arq in sorted(os.listdir(Features900)):
        path = os.path.join(Features900, arq)
        if(os.path.isfile(path)):
            l += 1
            print(arq)
            y, sr = lb.load(path, mono=True)

            mfccs = lb.feature.mfcc(y=y, sr=SR, n_mfcc=13)
            mfccs = np.apply_along_axis(stats, 1, mfccs).flatten()

            spectral_centroid = lb.feature.spectral_centroid(y=y, sr=SR)
            spectral_centroid = np.apply_along_axis(
                stats, 1, spectral_centroid).flatten()

            spectral_bandwith = lb.feature.spectral_bandwidth(y=y, sr=SR)
            spectral_bandwith = np.apply_along_axis(
                stats, 1, spectral_bandwith).flatten()

            spectral_contrast = lb.feature.spectral_contrast(y=y, sr=SR)
            spectral_contrast = np.apply_along_axis(
                stats, 1, spectral_contrast).flatten()

            spectral_flatness = lb.feature.spectral_flatness(y=y)
            spectral_flatness = np.apply_along_axis(
                stats, 1, spectral_flatness).flatten()

            spectral_rolloff = lb.feature.spectral_rolloff(y=y, sr=SR)
            spectral_rolloff = np.apply_along_axis(
                stats, 1, spectral_rolloff).flatten()

            min = 20
            max = 11025
            #fzero = lb.core.yin(y=y, fmin=min, fmax=max)
            fzero = lb.yin(y=y, fmin=min, fmax=max)
            fzero[fzero == max] = 0

            fzero = np.apply_along_axis(stats, 0, fzero).flatten()

            rms = lb.feature.rms(y=y)
            rms = np.apply_along_axis(stats, 1, rms).flatten()

            zero_crossing_rate = lb.feature.zero_crossing_rate(y=y)
            zero_crossing_rate = np.apply_along_axis(
                stats, 1, zero_crossing_rate).flatten()

            tempo = lb.beat.tempo(y=y, sr=SR)

            matrix[l] = np.concatenate((mfccs, spectral_centroid, spectral_bandwith, spectral_contrast,
                                       spectral_flatness, spectral_rolloff, fzero, rms, zero_crossing_rate, tempo))

    
    #print(mfccs.shape)
    #print(spectral_centroid.shape)
    #print(spectral_bandwith.shape)
    #print(spectral_contrast.shape)
    #print(spectral_flatness.shape)
    #print(spectral_rolloff.shape)
    #print(fzero.shape)
    #print(rms.shape)
    #print(zero_crossing_rate.shape)
    #print(tempo.shape)
    #print(matrix.shape)
    np.savetxt("Features - Audio MER\librosaNotNormalized0100.csv",
               matrix, delimiter=',', fmt="%.6f")
    np.savetxt("Features - Audio MER\librosaNormalized0100.csv", normaliza(
        matrix), delimiter=',', fmt="%.6f")


# Exercicio 3.1

def exercicio3_1():
    #dataLib = np.genfromtxt(
    #    'resultadosTP2/FMrosa.csv', delimiter=",")
    dataLib = np.genfromtxt(
        'Features - Audio MER\librosaNormalized0100.csv', delimiter=",")
    dataTop = np.genfromtxt(
        'Features - Audio MER/top100_features_normalized.csv', delimiter=",")
    if(Calcula3_1):
        euclidean_distance(dataLib, dataTop)
        manhattan_distance(dataLib, dataTop)
        cosseno_distance(dataLib, dataTop)


def euclidean_distance(dataLib, dataTop):
    linhas, colunas = np.shape(dataLib)

    DataFinalLib = np.zeros((900, 900), dtype=np.float64)

    DataFinalTop = np.zeros((900, 900), dtype=np.float64)

    for linha1 in range(linhas):
        for linha2 in range(linhas):
            if linha1 == linha2:
                DataFinalLib[linha1][linha2] = -1
                DataFinalTop[linha1][linha2] = -1

            else:
                DataFinalLib[linha1][linha2] = np.linalg.norm(
                    dataLib[linha1] - dataLib[linha2])
                DataFinalTop[linha1][linha2] = np.linalg.norm(
                    dataTop[linha1] - dataTop[linha2])

    np.savetxt("Features - Audio MER/EuclidianaLibrosa.csv",
               DataFinalLib, delimiter=',', fmt="%.6f")
    np.savetxt("Features - Audio MER/top100_features_normalized_euclidiana.csv",
               DataFinalTop, delimiter=',', fmt="%.6f")


def manhattan_distance(dataLib, dataTop):
    linhas, colunas = np.shape(dataLib)

    DataFinalLib = np.zeros((900, 900), dtype=np.float64)

    DataFinalTop = np.zeros((900, 900), dtype=np.float64)

    for linha1 in range(linhas):
        for linha2 in range(linhas):
            if linha1 == linha2:
                DataFinalLib[linha1][linha2] = -1
                DataFinalTop[linha1][linha2] = -1
            else:

                DataFinalLib[linha1][linha2] = np.sum(np.abs(
                    dataLib[linha1] - dataLib[linha2]))
                DataFinalTop[linha1][linha2] = np.sum(np.abs(
                    dataTop[linha1] - dataTop[linha2]))

    np.savetxt("Features - Audio MER/ManhattanLibrosa.csv",
               DataFinalLib, delimiter=',', fmt="%.6f")
    np.savetxt("Features - Audio MER/top100_features_normalized_manhattan.csv",
               DataFinalTop, delimiter=',', fmt="%.6f")



def cosseno_distance(dataLib, dataTop):
    linhas, colunas = np.shape(dataLib)

    DataFinalLib = np.zeros((900, 900), dtype=np.float64)

    DataFinalTop = np.zeros((900, 900), dtype=np.float64)

    for linha1 in range(linhas):
        for linha2 in range(linhas):
            if linha1 == linha2:
                DataFinalLib[linha1][linha2] = -1
                DataFinalTop[linha1][linha2] = -1
            else:
                DataFinalLib[linha1][linha2] = 1 - np.dot(dataLib[linha1], dataLib[linha2]) / (
                    np.linalg.norm(dataLib[linha1])*np.linalg.norm(dataLib[linha2]))

                DataFinalTop[linha1][linha2] = 1 - np.dot(dataTop[linha1], dataTop[linha2]) / (
                    np.linalg.norm(dataTop[linha1])*np.linalg.norm(dataTop[linha2]))

    np.savetxt("Features - Audio MER/CossenoLibrosa.csv",
               DataFinalLib, delimiter=',', fmt="%.6f")
    np.savetxt("Features - Audio MER/top100_features_normalized_cosseno.csv",
               DataFinalTop, delimiter=',', fmt="%.6f")
    


def build_matrix(musica):
    file = 'MER_audio_taffc_dataset/panda_dataset_taffc_metadata.csv'
    data = np.genfromtxt(file, dtype=np.str_,delimiter=",")

    res = np.where(musica == data[:,0])[0][0]
    
    print("Query " + musica)

    #Euclidiana
    DataFinalLib = np.genfromtxt("Features - Audio MER/EuclidianaLibrosa.csv", dtype=np.str_,delimiter=",")
    #DataFinalLib = np.genfromtxt("resultadosTP2\der.csv", dtype=np.str_,delimiter=",")

    DataFinalTop = np.genfromtxt("Features - Audio MER/top100_features_normalized_euclidiana.csv", dtype=np.str,delimiter=",")

    librosaValues = DataFinalLib[res-1]
    librosaValues = np.argsort(librosaValues)

    top100Values = DataFinalTop[res-1]
    top100Values = np.argsort(top100Values)

    finalLibrosa = [data[librosaValues[i]+1][0] for i in range(21)]
    finaltop100 = [data[top100Values[i]+1][0] for i in range(21)]
    print("Euclidean Librosa:")
    print(finalLibrosa)
    print("Euclidean Top 100:")
    print(finaltop100)
    print("\n")

    #Manhathan
    DataFinalLib = np.genfromtxt("Features - Audio MER\ManhattanLibrosa.csv", dtype=np.str_,delimiter=",")
    DataFinalTop = np.genfromtxt("Features - Audio MER/top100_features_normalized_manhattan.csv", dtype=np.str,delimiter=",")

    librosaValues = DataFinalLib[res-1]
    librosaValues = np.argsort(librosaValues)

    top100Values = DataFinalTop[res-1]
    top100Values = np.argsort(top100Values)

    finalLibrosa = [data[librosaValues[i]+1][0] for i in range(21)]
    finaltop100 = [data[top100Values[i]+1][0] for i in range(21)]
    print("Manhatan Librosa:")
    print(finalLibrosa)
    print("Manhatan Top 100:")
    print(finaltop100)
    print("\n")

    #Cosseno

    DataFinalLib = np.genfromtxt("Features - Audio MER\CossenoLibrosa.csv", dtype=np.str_,delimiter=",")
    DataFinalTop = np.genfromtxt("Features - Audio MER/top100_features_normalized_cosseno.csv", dtype=np.str,delimiter=",")

    librosaValues = DataFinalLib[res-1]
    librosaValues = np.argsort(librosaValues)

    top100Values = DataFinalTop[res-1]
    top100Values = np.argsort(top100Values)

    finalLibrosa = [data[librosaValues[i]+1][0] for i in range(21)]
    finaltop100 = [data[top100Values[i]+1][0] for i in range(21)]
    print("Cosine Librosa:")
    print(finalLibrosa)
    print("Cosine Top 100:")
    print(finaltop100)
    print("\n")


def main():
    #data1 = np.genfromtxt('resultadosTP2/FMrosa.csv', delimiter=",")
    #data2 = np.genfromtxt('Features - Audio MER\librosaNormalized0100.csv', delimiter=",")
    #print(np.max(np.abs((np.subtract(data1,data2)))))
    #print(np.where(np.max(np.abs(np.subtract(data1,data2))) == np.abs(np.subtract(data1,data2))))
    #print(np.max(np.abs(np.subtract(data1[359],data2[359]))))
    #array = np.where(np.max(np.abs(np.subtract(data1[359],data2[359])))==np.max(np.abs(np.subtract(data1,data2))))
    #print(array)
    #for i in range(190):
    #    if(np.abs(data1[359][i]-data2[359][i])==0.16059): print(i)
    #print(data2[359][101])
    #print(data1[359][101])

##################################################################################################
    warnings.filterwarnings("ignore")
    # read_csv()
    #extractLibrosa2_1_1()
    #exercicio3_1()
    orig_stdout = sys.stdout
    sys.stdout = open('Features - Audio MER/rankings.txt', 'w')
    for arq in sorted(os.listdir("Queries")):
        
        file = arq[:-4]
        build_matrix('"'+file+'"')
        print("\n\n")
    sys.stdout = orig_stdout


if __name__ == "__main__":

    main()
