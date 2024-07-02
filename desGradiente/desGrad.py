# Regresión lineal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit

# Leer datos
data = pd.read_csv('data.csv')
X = data.iloc[:, 0]
Y = data.iloc[:, 1]

# Mínimos cuadrados
N = len(X)  
sumx = sum(X)
sumy = sum(Y)
sumxy = sum(X * Y)
sumx2 = sum(X * X)
w1 = (N * sumxy - sumx * sumy) / (N * sumx2 - sumx * sumx)
w0 = (sumy - w1 * sumx) / N
Ybar = w0 + w1 * X

# Descenso de gradiente
w0 = 0.0
w1 = 0.0
alpha = 0.025
epocs = 100

# Definición de la función de descenso de gradiente
@jit(nopython=True)
def descensoG(epocs, sumx, sumy, sumxy, sumx2, N, alpha):
    w0 = 0.0
    w1 = 0.0
    for i in range(epocs):
        Gradw0 = -2.0 * (sumy - w0 * N - w1 * sumx)
        Gradw1 = -2.0 * (sumxy - w0 * sumx - w1 * sumx2)
        w0 -= alpha * Gradw0
        w1 -= alpha * Gradw1
    return w0, w1

w0, w1 = descensoG(epocs, sumx, sumy, sumxy, sumx2, N, alpha)
Ybar2 = w0 + w1 * X

# Gráfica
plt.scatter(X, Y)
plt.rcParams['figure.figsize'] = (12.0, 9.0)
plt.plot([min(X), max(X)], [min(Ybar), max(Ybar)], color='red', label='Mínimos Cuadrados')
plt.plot([min(X), max(X)], [min(Ybar2), max(Ybar2)], color='green', label='Descenso de Gradiente')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
