### Independent Components Analysis
###
### This program requires a working installation of:
###
### On Mac:
###     1. portaudio: On Mac: brew install portaudio
###     2. sounddevice: pip install sounddevice
###
### On windows:
###      pip install pyaudio sounddevice
###

import sounddevice as sd
import numpy as np
import os, random
Fs = 11025

def normalize(dat):
    M, N = dat.shape
    normdat=np.zeros((M,N))
    for i in range(M):
        for j in range(N):
            normdat[i,j]=0.99*dat[i,j]/np.max(np.abs(dat[:,j]))
    return normdat

def load_data():
    mix = np.loadtxt('mix.dat')
    return mix

def play(vec):
    sd.play(vec, Fs, blocking=True)

def unmixer(X):
    M, N = X.shape
    W = np.eye(N)

    anneal = [0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01, 0.01,
              0.005, 0.005, 0.002, 0.002, 0.001, 0.001]
    print('Separating tracks ...')
    for i in range(len(anneal)):
        for j in range(M):
            randrow=X[j,:]
            sigm=1/(1+np.exp(-np.matmul(W,randrow)))
            W=W+anneal[i]*(np.outer((1-2*sigm),randrow)+np.linalg.inv(np.transpose(W)))
    return W

def unmix(X, W):
    S = np.zeros(X.shape)
    S=np.inner(X,W)

    return S

def main():
    X = normalize(load_data())

    for i in range(X.shape[1]):
        print('Playing mixed track %d' % i)
        play(X[:, i])

    W = unmixer(X)
    S = normalize(unmix(X, W))

    for i in range(S.shape[1]):
        print('Playing separated track %d' % i)
        play(S[:, i])

if __name__ == '__main__':
    main()
