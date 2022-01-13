from numpy import *
import matplotlib.pyplot as plt

x = arange(-4, 30, 0.01)

def w_inf(a, b, t):
    f = (1 / a ** 0.5) * exp(-0.5 * ((t - b) / a) ** 2) * (((t - b) / a) ** 2 - 1)
    return f

def gabor_inf(a, b, t):
    f = (1 / a ** 0.5) * exp(-0.5 * ((t - b) / a) ** 2) * cos(5 * (t - b) / a)
    return f

plt.title("Вейвлет «Мексиканская шляпа»:\n$1/\sqrt{a}*exp(-0,5*t^{2}/a^{2})*(t^{2}-1)$")
y = [w_inf(1, 12, t) for t in x]
y2 = [gabor_inf(1, 12, t) for t in x]
plt.plot(x, y, label="$\psi(t)$ a=1,b=12 hat")
plt.plot(x, y2, label="$\psi(t)$ a=1,b=12 gabor")

plt.legend(loc='best')
plt.grid(True)
plt.show()


def plotSpectrum(y, Fs):
    n = len(y)
    k = arange(n)
    T = n / Fs
    frq = k / T
    frq = frq[range(int(n / 2))]
    Y = fft.fft(y) / n
    Y = Y[range(int(n / 2))]
    return Y, frq


Fs = 1024.0
y = [w_inf(1, 12, t) for t in x]
y2 = [gabor_inf(1, 12, t) for t in x]

Y, frq = plotSpectrum(y, Fs)
plt.plot(frq, abs(Y), label="$\psi(\omega)$ a=1,b=12 hat")
Y, frq = plotSpectrum(y2, Fs)
plt.plot(frq, abs(Y), label="$\psi(\omega)$ a=1,b=12 gabor")
plt.legend(loc='best')
plt.grid(True)
plt.show()

from scipy.integrate import quad
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

N = 256


def S(t):
    # if(t<30):
    #    return sin(2*pi*t/10)
    # return sin(2*pi*t/50)
    return t + 5 * sin(2 * pi * t / 10)


plt.figure()
plt.title(' Гармоническое колебание', size=12)
y = [S(t) for t in arange(0, 100, 1)]
x = [t for t in arange(0, 100, 1)]
plt.plot(x, y)
plt.grid()
plt.show()


def w(a, b):
    f = lambda t: (1 / a ** 0.5) * exp(-0.5 * ((t - b) / a) ** 2) * (((t - b) / a) ** 2 - 1) * S(t)
    r = quad(f, -N, N)
    return round(r[0], 3)


def gabor_w(a, b):
    f = lambda t: (1 / a ** 0.5) * exp(-0.5 * ((t - b) / a) ** 2) * cos(5 * (t - b) / a) * S(t)
    r = quad(f, -N, N)
    return round(r[0], 3)


x = arange(1, 50, 1)
y = arange(1, 50, 1)
z = array([w(i, j) for j in y for i in x])
X, Y = meshgrid(x, y)
Z = z.reshape(49, 49)
fig = plt.figure("Вейвлет- спектр: гармонического колебания")
ax = Axes3D(fig)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet)
ax.set_xlabel(' Масштаб:a')
ax.set_ylabel('Задержка: b')
ax.set_zlabel('Амплитуда ВП: $ N_{ab}$')
# plt.figure("2D-график для z = w (a,b)")
# plt.title('Плоскость ab с цветовыми областями ВП', size=12)
# plt.contourf(X, Y, Z,100)
plt.show()

x = arange(1, 50, 1)
y = arange(1, 50, 1)
z = array([gabor_w(i, j) for j in y for i in x])
X, Y = meshgrid(x, y)
Z = z.reshape(49, 49)
fig = plt.figure("Вейвлет- спектр: гармонического колебания")
ax = Axes3D(fig)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet)
ax.set_xlabel(' Масштаб:a')
ax.set_ylabel('Задержка: b')
ax.set_zlabel('Амплитуда ВП: $ N_{ab}$')
# plt.figure("2D-график для z = w (a,b)")
# plt.title('Плоскость ab с цветовыми областями ВП', size=12)
# plt.contourf(X, Y, Z,100)
plt.show()
