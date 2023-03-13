from numpy import exp, sin, cos, log, pi, sign
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import solve_ivp
from scipy.io import loadmat
from scipy.spatial import ConvexHull
import csv

b1 = np.loadtxt('b_sse.csv',delimiter=",")
b2 = np.loadtxt('b_sse2.csv',delimiter=",")
q1 = np.loadtxt('qr_sse.csv',delimiter=",")
q2 = np.loadtxt('qr_sse2.csv',delimiter=",")

xgrid = np.loadtxt('xval.csv',delimiter=",")
vgrid = np.loadtxt('vval.csv',delimiter=",")
# print(b1)

bdiff = np.abs(b1 - b2)
qdiff = np.abs(q1 - q2)

fig, ax = plt.subplots()
c = ax.pcolormesh(xgrid, vgrid, bdiff, vmin= 0, vmax = 0.05)
fig.colorbar(c, ax=ax, label='Squared Error')
ax.set(xlabel='Angle',ylabel='Angular Velocity')
plt.title("Squared Error Difference - EDMD")
plt.show()


fig, ax = plt.subplots()
c = ax.pcolormesh(xgrid, vgrid, qdiff, vmin= 0, vmax = 0.05)
fig.colorbar(c, ax=ax, label='Squared Error')
ax.set(xlabel='Angle',ylabel='Angular Velocity')
plt.title("Squared Error Difference - DDE")
plt.show()
