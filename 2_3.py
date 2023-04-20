import scipy as sp
import numpy as np

# parametros default: sr = 22050 Hz, mono, 
# window length = frame length = 92.88 ms || 2048 e hop length = 23.22 ms || 512

hop_length = 512
frame_length = 2048

def mfcc(signal,windows,start,end):
    array = np.zeros((1,windows))  
    return array


# Centre of gravity of the magnitude spectrum (DFT), in Hz
def spectral_centroid(signal,windows,start,end):
    array = np.zeros((1,windows))  
    
    return array
    


def spectral_bandwith(signal,windows,start,end):
    array = np.zeros((1,windows))  

def spectral_flatness(signal,windows,start,end):
    array = np.zeros((1,windows))  
    return array

def spectral_rolloff(signal,windows,start,end):
    array = np.zeros((1,windows))  
    return array


# Measure the energy of a signal over a window
def rms(signal,windows,start,end):
    array = np.zeros((1,windows))   
    for i in range(windows):
        # Xrms = sqr((1/N)sum(de n=1 a N)(x[n]^2))
        array[0,i] =np.sqrt( np.sum(signal[start[i]:end[i]]**2)/(frame_length))
    return array



#The rate at which a signal changes from positive to negative or from negative to positive
def zero_crossing_rate(signal,windows,start,end):
    array = np.zeros((1,windows))
    # zcr = (1/2N) sum(de n=2 a N)( |sinal[x[n]]-sinal[x[n-1]]| )
    for i in range(windows):
        array[0,i] = np.sum(np.abs(np.diff(signal[start[i]:end[i]] > 0)))/(frame_length)
    return array




def calculaStats(signal):
    size = len(signal)
    windows = (size) // hop_length + 1

    start= np.arange(windows)*hop_length - frame_length//2
    start[start<0]=0
    end= np.arange(windows)*hop_length + frame_length//2
    end[end>size]= size

    mfcc_array = mfcc(signal,windows,start,end)
    spectral_centroid_array = spectral_centroid(signal,windows,start,end)
    spectral_flatness_array = spectral_flatness(signal,windows,start,end)
    spectral_rolloff_array = spectral_rolloff(signal,windows,start,end)
    rms_array = rms(signal,windows,start,end)
    zcr_array = zero_crossing_rate(signal,windows,start,end)






def main():
    pass

if __name__ == "__main__":
    main()