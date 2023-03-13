# %%
import sys
sys.path.append('../')
import pandas as pd
import numpy as np
from numpy import exp, sin, cos, log, pi
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# %% functions

def RBF(x, c, epsilon, kinds="gauss"):
	# print('x:' + str(x))
	# print('c:' + str(c))
	r = np.linalg.norm(x - c)
	# print('r:' + str(r))
	if kinds == "gauss":
		return exp(-(epsilon*r)**2)
	elif kinds == "quad":
		return 1/(1+(epsilon*r)**2)
	else:
		pass

def dot_product(x, y):
	return np.dot(x, y)



if __name__ =="__main__":


	n_c = 5
	N_states = 2
	datasizes = [625, 900, 2500, 10000, 22500]
	for dsize in datasizes:

		d_t = pd.read_csv('./data/agg_t_'+str(dsize)+ '.csv', header=None)
		d_t1 = pd.read_csv('./data/agg_t1_'+str(dsize)+ '.csv', header=None)
		data_t = d_t.values
		data_t1 = d_t1.values
		data = {'x': {
			'minus': data_t,
			'plus': data_t1
		}}
		x_minus = data['x'  ]['minus']
		x_plus  = data['x'  ]['plus' ]

		rbf_min = np.ndarray.min(x_minus,0)
		rbf_max = np.ndarray.max(x_minus,0)
		rbf_c1 = np.linspace(rbf_min[0], rbf_max[0], n_c, dtype="float64")
		rbf_c2 = np.linspace(rbf_min[1], rbf_max[1], n_c,dtype="float64")
		rbf_c = np.array(np.meshgrid(rbf_c1, rbf_c2)).T.reshape(-1, N_states)

		rbf_dilation = np.ones(rbf_c.shape,dtype="float64")

		g = []
		g_t1 = []

		for i in range(0, len(x_minus)):
			temp_t = []
			temp_t1 = []
			temp_t= x_minus[i,:].tolist()
			temp_t1= x_plus[i,:].tolist()
			for j in range(0, len(rbf_c)):
				temp_t.append(RBF(x_minus[i,:], rbf_c[j,:], rbf_dilation[j,0]))
				temp_t1.append(RBF(x_plus[i,:], rbf_c[j,:], rbf_dilation[j,0]))

			g.append(temp_t)
			g_t1.append(temp_t1)

		
		gX = np.matrix(g)
		gFx = np.matrix(g_t1)
		gX_T = gX.transpose()
		gFx_T = gFx.transpose()

		A_traj = gFx_T@np.linalg.pinv(gX_T)

		base_str = './models/base' + str(n_c) +'_' + str(dsize)+ '.csv'
		np.savetxt(base_str, A_traj, delimiter=",")

