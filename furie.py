import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


def myDFT(x):
    # x - 1D array
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)


def myDFT_inv(x):
    x = np.asarray(x)
    N = x.shape[0]
    k = np.arange(N)
    n = k.reshape((N, 1))
    M = np.exp(2j * np.pi * k * n / N) / N
    return np.dot(M, x)


def plot_signal_furie(x, n):
    fig, axes = plt.subplots(nrows=2, ncols=4,
                             figsize=(16, 8)
                             )

    axes[0, 0].plot(x)
    y = np.fft.fft(x)
    y2 = myDFT(x)

    axes[0, 1].set(xlim=(0, n / 2))
    axes[0, 2].set(xlim=(0, n / 2))
    axes[0, 3].set(xlim=(0, n / 2))
    axes[1, 3].set(xlim=(0, n / 2))
    axes[0, 1].plot(np.absolute(y2), color=(0, 0, 0.5))  # blue
    axes[0, 2].plot(np.absolute(y), color=(0.5, 0, 0))  # red
    axes[1, 0].plot(np.absolute(myDFT(x) - y))
    axes[1, 1].plot(myDFT_inv(y2), color=(0, 0, 0.5))
    axes[1, 2].plot(np.fft.ifft(y), color=(0.5, 0, 0))
    axes[0, 3].plot(y2.imag, color=(0, 0, 0.5))
    axes[1, 3].plot(y.imag, color=(0.5, 0, 0))
    plt.show()
