# Regresión lineal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# Gráfica
plt.scatter(X, Y)
plt.rcParams['figure.figsize'] = (12.0, 9.0)
plt.plot([min(X), max(X)], [min(Ybar), max(Ybar)], color='red')
plt.xlabel("x")
plt.ylabel("y")
plt.show()
