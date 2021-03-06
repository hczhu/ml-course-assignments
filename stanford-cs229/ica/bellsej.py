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
# Algorithm: http://cs229.stanford.edu/notes/cs229-notes11.pdf

import sounddevice as sd
import numpy as np

Fs = 11025

def normalize(dat):
    return 0.99 * dat / np.max(np.abs(dat))

def load_data():
    mix = np.loadtxt('data/mix.dat')
    print(mix.shape)
    return mix

def play(vec):
    sd.play(vec, Fs, blocking=True)

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x.clip(-100, 100)))

def unmixer(X):
    M, N = X.shape
    W = np.eye(N, dtype=float)

    anneal = [0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01, 0.01,
              0.005, 0.005, 0.002, 0.002, 0.001, 0.001]
    print('Separating tracks ...')
    ######## Your code here ##########
    for epoch in range(len(anneal)):
        lr = anneal[epoch]
        for i in range(M):
            W += lr * (np.linalg.inv(W.T) + np.outer(1 - 2 * sigmoid(W.dot(X[i, :])), X[i, :]))
    ###################################
    return W

def unmix(X, W):
    # S = np.zeros(X.shape)
    return np.matmul(X, W.T)

def main():
    X = normalize(load_data())

    for i in range(X.shape[1]):
        print('Playing mixed track %d' % i)
        # play(X[:, i])

    W = unmixer(X)
    S = normalize(unmix(X, W))

    for i in range(S.shape[1]):
        print('Playing separated track %d' % i)
        play(S[:, i])

if __name__ == '__main__':
    main()
