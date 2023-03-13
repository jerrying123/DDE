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
df = pd.read_csv('./base.csv', header=None)
error_b = np.array(df)
df = pd.read_csv('./dde.csv', header=None)
error_dde = np.array(df)

bmean = np.mean(error_b,axis=0)
bvarm = np.std(error_b, axis=0)
ddemean = np.mean(error_dde, axis=0)
ddevarm = np.std(error_dde, axis=0)

print('EDMD Means: ', bmean)
print('EDMD Vars: ', bvarm)

print('DDE Means: ', ddemean)
print('DDE Vars: ', ddevarm)