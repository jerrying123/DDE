# %%
import sys
sys.path.append('../')
import pandas as pd
import numpy as np
from numpy import exp, sin, cos, log, pi
import matplotlib.pyplot as plt
import os
from scipy.spatial import ConvexHull, Delaunay
import itertools
import matplotlib.pyplot as plt
import matplotlib
df = pd.read_csv('./q.csv', header=None)
print(df)

data = np.array(df)
print(data)
fig,ax = plt.subplots()
ax.plot(data[:,0], data[:,1],'--o')
ax.plot(data[:,0], data[:,2],'-*')
ax.plot(data[:,0], data[:,3],'-.o')
ax.plot(data[:,0], data[:,4],'-o')
ax.set_xscale('log')
ax.set_xticks([200, 500, 1000, 5000])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_xlabel('Dataset size')
ax.set_ylabel('Parameter value')
ax.legend(['$Q_{0,0}$', '$Q_{3,3}$', '$Q_{5,5}$','$Q_{13,13}$'])
plt.savefig('C:/Users/Jerry/Desktop/assorted figures/dde/pendulum/q_plot.png')
plt.show()