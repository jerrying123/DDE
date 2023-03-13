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
df = pd.read_csv('./q.csv', header=None)
print(df)

data = np.array(df)
print(data)
fig,ax = plt.subplots()
ax.plot(data[:-1,0], data[:-1,1],'-o')
ax.plot(data[:-1,0], data[:-1,2],'-o')
ax.plot(data[:-1,0], data[:-1,3],'-o')
ax.plot(data[:-1,0], data[:-1,4],'-o')
ax.set_xticks(data[2:-1,0])
ax.set_xlabel('Dataset size')
ax.set_ylabel('Parameter value')
ax.legend(['$Q_{0,1}$', '$Q_{3,4}$', '$Q_{5,7}$','$Q_{13,17}$'])
plt.savefig('C:/Users/Jerry/Desktop/assorted figures/dde/pendulum/q_plot_gaussian.png')
plt.show()