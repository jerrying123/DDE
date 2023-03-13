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


a = [27, 51, 83]
nums = np.array([1,2,3])
b = [31.69, 36.657, 28.437]
c = [25.099, 21.637, 13.613]
width = 0.2
plt.bar(nums-0.1, b, width, color='cyan')
plt.bar(nums+0.1, c, width, color='green')
plt.xticks(nums,a)
plt.xlabel("Number of Observables")
plt.ylabel("Total SSE over Dynamic Range")
plt.legend(['EDMD', 'DDE'])
plt.show()