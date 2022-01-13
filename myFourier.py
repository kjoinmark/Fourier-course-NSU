import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import signal
from furie import *


def convolution(x1, x2):
    n, m = len(x1), len(x2)
    x1 = np.concatenate([np.zeros(m - 1, dtype=int), x1])
    x2 = np.concatenate([x2[::-1], np.zeros(n - 1, dtype=int)])
    return np.array([np.sum(x1 * np.roll(x2, i)) for i in range(m + n - 1)])


def cross_corelation(x1, x2):
    return convolution(x1, np.flip(x2))


def plot_signal_corr(y, y1, x):
    fig, axes = plt.subplots(nrows=1, ncols=4,
                             figsize=(10, 5)
                             )

    axes[0].plot(x, y)
    axes[1].plot(x, y1)

    axes[2].plot(y, y1, convolution(y, y1), 'black')
    axes[2].plot(y, y1, np.convolve(y, y1, mode='full'), 'g')

    axes[3].plot(y, y1, cross_corelation(y, y1), 'black')
    axes[3].plot(y, y1, np.correlate(y, y1, "full"), 'g')
    plt.show()


h_end_long = lambda x: [1 if i > 11 and i < 19 else 0 for i in x]
h_begin_long = lambda x: [1 if i > 1 and i < 10 else 0 for i in x]
h_mid_long = lambda x: [1 if i > 6 and i < 15 else 0 for i in x]
h_short = lambda x: [1 if i > 3 and i < 6 else 0 for i in x]
h_doubled = lambda x: [1 if i > 1 and i < 5 or i > 10 and i < 15 else 0 for i in x]
sin_x = np.sin(np.linspace(0, np.pi * 2, 10000))
sin_2x = np.sin(10 * np.linspace(0, np.pi * 2, 10000))
noize1 = [random.uniform(-1, 1) * 0.01 for i in range(1000)]
noize2 = [random.uniform(-1, 1) * 0.01 for i in range(1000)]
x = np.linspace(0, 20, 10000)
x2 = np.linspace(0, 40, 10000)

x1 = np.linspace(0, 1000, 1000)

plt.ion()
y = h_end_long(x)
y2 = h_begin_long(x2)
fig, axes = plt.subplots(nrows=1, ncols=4,
                         figsize=(10, 5))
axes[0].plot(x, y)
axes[1].plot(x2, y2)

axes[2].plot(y, y2, convolution(y, y2), 'black')
axes[2].plot(y, y2, np.convolve(y, y2, mode='full'), 'g')

axes[3].plot(y, y2, cross_corelation(y, y2), 'black')
axes[3].plot(y, y2, np.correlate(y, y2, "full"), 'g')
plt.show()

plot_signal_corr(h_end_long(x), h_end_long(x), x)
plot_signal_corr(h_end_long(x), h_begin_long(x), x)
plot_signal_corr(h_begin_long(x), h_mid_long(x), x)
plot_signal_corr(h_begin_long(x), h_short(x), x)
plot_signal_corr(sin_x, sin_x, x)
plot_signal_corr(sin_x, sin_2x, x)
plt.ioff()
plot_signal_corr(noize1, noize2, x1)
