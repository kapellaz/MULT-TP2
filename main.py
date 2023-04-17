"""
Bruno Sequeira: 2020235721
Rui Santos: 2020225542
Tomás Dias: 2020215701
"""

import numpy as np
import librosa as lb
import os
import scipy.stats as scp

SR = 22050


Features900 = "MER_audio_taffc_dataset\FeaturesQuadrantes"


#Exercicio 1
def stats(array):
    mean = np.mean(array)
    std = np.std(array)
    skewness = scp.skew(array)
    kurtosis = scp.kurtosis(array)
    media = np.median(array)
    max = np.max(array)
    min = np.min(array)

    return np.array([mean, std, skewness,kurtosis,media,max,min])



def read_csv():

    data = np.genfromtxt(
        'Features - Audio MER/top100_features.csv', dtype=np.str_, delimiter=",")

    linhas, colunas = np.shape(data)

    fNames = data[0, 1:colunas - 1]

    data = data[1::, 1:(colunas - 1)].astype(float)

    dataNormalizada = normaliza(data)

    np.savetxt("Features - Audio MER/top100_features_normalized.csv",dataNormalizada, fmt="%lf", delimiter=",")



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





#Exercicio 2
def extractLibrosa2_1_1():
    #features = np.zeros(900,190)

    matrix = np.zeros((900,190), dtype=np.float64)

    l = -1
    for arq in sorted(os.listdir(Features900)):
        path = os.path.join(Features900,arq)
        if(os.path.isfile(path)): 
            l+=1
            print(arq)
            y, sr = lb.load(path,mono=True)

            mfccs = lb.feature.mfcc(y=y, sr=SR,n_mfcc=13)
            mfccs = np.apply_along_axis(stats,1,mfccs).flatten()

            spectral_centroid = lb.feature.spectral_centroid(y=y,sr=SR)
            spectral_centroid = np.apply_along_axis(stats,1,spectral_centroid).flatten()

            spectral_bandwith = lb.feature.spectral_bandwidth(y=y,sr=SR)
            spectral_bandwith = np.apply_along_axis(stats,1,spectral_bandwith).flatten()


            spectral_contrast = lb.feature.spectral_contrast(y=y,sr=SR)
            spectral_contrast = np.apply_along_axis(stats,1,spectral_contrast).flatten()



            spectral_flatness  = lb.feature.spectral_flatness(y=y)
            spectral_flatness = np.apply_along_axis(stats,1,spectral_flatness).flatten()

            spectral_rolloff = lb.feature.spectral_rolloff(y=y,sr=SR)
            spectral_rolloff = np.apply_along_axis(stats,1,spectral_rolloff).flatten()

            min = 20
            max = 11205
            fzero= lb.core.yin(y=y,fmin=min,fmax=max)
            fzero[fzero == max]= 0

            fzero = np.apply_along_axis(stats,0,fzero).flatten()


            rms = lb.feature.rms(y=y)
            rms = np.apply_along_axis(stats,1,rms).flatten()

            zero_crossing_rate = lb.feature.zero_crossing_rate(y=y)
            zero_crossing_rate = np.apply_along_axis(stats,1,zero_crossing_rate).flatten()

            
            tempo = lb.beat.tempo(y=y,sr=SR)

            matrix[l] = np.concatenate((mfccs,spectral_centroid,spectral_bandwith,spectral_contrast,spectral_flatness,spectral_rolloff,fzero,rms,zero_crossing_rate,tempo))

    print(matrix.shape)
    np.savetxt("librosaNotNormalized0100.csv", matrix,delimiter=',',fmt="%.6f")
    np.savetxt("librosaNormalized0100.csv",normaliza(matrix),delimiter=',',fmt="%.6f")
 



def main():
    #read_csv()
    #extractLibrosa2_1_1()
    data = np.genfromtxt('resultadosStor\FMrosa.csv', delimiter=",")

    
    

if __name__ == "__main__":
    main()
    