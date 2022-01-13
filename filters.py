import numpy as np
import matplotlib.pyplot as plt

n_data = 1000
cut_freq = 80
accuracy = 0.005


def low_idle(cfreq, n):
    idle_arr = np.zeros(n, dtype=complex)
    for i in range(0, cfreq):
        idle_arr[i] = complex(1., 0.)
        idle_arr[n - i - 1] = complex(1., 0.)
    res = np.fft.ifft(idle_arr)
    res = res.real
    for i in range(0, res.size):
        if (-accuracy < res[i] < accuracy):
            res[i] = 0
    res = np.fft.fft(res)
    return res


def high_idle(cfreq, n):
    idle_arr = np.zeros(n, dtype=complex)
    for i in range(cfreq, n - 1 - cfreq):
        idle_arr[i] = complex(1., 0.)
    res = np.fft.ifft(idle_arr)
    res = res.real
    for i in range(0, res.size):
        if (-accuracy < res[i] < accuracy):
            res[i] = 0
    res = np.fft.fft(res)
    return res


def low_freq_filter(data, cfreq):
    idle_arr = low_idle(cfreq, data.size)
    f_data = np.fft.fft(data)
    res = np.multiply(f_data, idle_arr)
    res = np.fft.ifft(res)
    res = res.real
    return res


def high_freq_filter(data, cfreq):
    idle_arr = high_idle(cfreq, data.size)
    f_data = np.fft.fft(data)
    res = np.multiply(f_data, idle_arr)
    res = np.fft.ifft(res)
    res = res.real
    return res


def narrow_band_filter(data, lowfreq, highfreq):
    res = low_freq_filter(data, highfreq)
    res = high_freq_filter(res, lowfreq)

    return res


def notch_filter(data, lowfreq, highfreq):
    res = low_freq_filter(data, lowfreq)
    res2 = high_freq_filter(data, highfreq)
    res += res2

    return res


def band_stop_filter(data, lowfreq, highfreq):
    res = narrow_band_filter(data, lowfreq, highfreq)
    res = data - res
    return res


def oversampling_FFT(signal, n, m):  # n - начальное m - конечное число отсчетов
    fft = np.fft.fft(signal)
    furie = np.zeros(m, dtype=complex)
    for i in range(n):
        furie[i] = fft[i]
    if (n % 2):
        half_n = int((n + 1) / 2)
        for i in range(half_n + m - n + 1, m):
            furie[i] = furie[i - m + n]
        for i in range(half_n, half_n + m - n):
            furie[i] = 0
    else:
        half_n = int(n / 2)
        for i in range(half_n + m - n + 2, m):
            furie[i] = furie[i - m + n]
        furie[half_n + m - n + 1] = furie[half_n + 1] / 2
        furie[half_n + 1] = furie[half_n + 1] / 2
        for i in range(half_n + 2, half_n + m - n):
            furie[i] = 0
    res = np.fft.ifft(furie)
    res = res.real
    res = res * m / n
    return res


def decimation(signal, n, m):  # n - начальное m - во сколько раз уменьшаем
    cut = int(n / m / 2)
    data = low_freq_filter(signal, cut)
    res = np.zeros(int(n / m))
    j = 0
    for i in range(0, n - m + 1, m):
        res[j] = data[i]
        j += 1
    return res


def samplerating(signal, n, k=1, m=1):  # n - начальное m - во сколько раз уменьшаем k - во сколько увеличиваем
    if (k / m > 1):
        res = oversampling_FFT(signal, n, int(n * k / m))
        return res
    else:
        res = oversampling_FFT(signal, n, int(n * k))
        print(res.size)
        res = decimation(res, n * k, m)
        return res


x = np.random.rand(n_data)
for i in range(n_data):
    x[i] += 10 * np.sin(i / 50) + 5 * np.sin(i * 2)

res = oversampling_FFT(x, n_data, n_data * 2)
res2 = decimation(x, n_data, 3)
res3 = samplerating(x, n_data, 4, 1)

fig, axes = plt.subplots(nrows=2, ncols=4,
                         figsize=(15, 5)
                         )
axes[0, 0].plot(x)
axes[1, 0].plot(np.abs(np.fft.fft(x)))

axes[0, 1].plot(res)
axes[1, 1].plot(np.abs(np.fft.fft(res)))

axes[0, 2].plot(res2)
axes[1, 2].plot(np.abs(np.fft.fft(res2)))

axes[0, 3].plot(res3)
axes[1, 3].plot(np.abs(np.fft.fft(res3)))

plt.show()

axis = np.linspace(0, n_data, num=res3.size)
plt.plot(axis, res3, 'bo')
plt.plot(x, 'ro')
plt.show()
