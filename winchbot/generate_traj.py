#!/usr/bin/env python
from numpy import exp, sin, cos, log, pi, sign

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import solve_ivp
from scipy.io import savemat
import helpers.learn_modules as lm
from helpers.networkarch import NeuralNetwork
import torch
# %% functions
def RBF(x, c, epsilon, xind, kinds="gauss"):
	# print('x:' + str(x))
	# print('c:' + str(c))
	# print('e:' + str(epsilon))
	# print('i:' + str(xind))

	r = np.linalg.norm(x[xind] - c)
	# print('r:' + str(r))
	if kinds == "gauss":
		return exp(-(epsilon*r)**2)
	elif kinds == "quad":
		return 1/(1+(epsilon*r)**2)
	else:
		pass


def diffeq(t, y):

	gain = 19
	x0 = y
	xw1 = [0,0]
	xw2 = [2,0]
	k = 1699
	r = 0.3
	I = 0.45
	g = - 9.8
	m = 10
	xf = [0.8, -4, 0, 0, 4.1145, 0]
	u = -gain*(xf[4] - x0[4]) + x0[5]
	d1 = calc_d(x0, xw1)
	h = find_eta(x0, m, xw1, xw2, k)
	dxdt = np.zeros(6)
	dxdt[0] = x0[2]
	dxdt[1] = x0[3]
	dxdt[2] = (h[0])/m#; % x acceleration
	dxdt[3] = (h[1]  + m*g)/m#; % y acceleration
	dxdt[4] = x0[5]#; % winch cable a velocity
	if np.linalg.norm(d1) < x0[4]:
	    dxdt[5] = -r/I*(u)
	else:
	    dxdt[5] = -r/I*(u - r * np.sqrt(h[0]**2+h[1]**2)); 


	return dxdt
def calc_d(x, xw):

	dx = xw[0] - x[0]
	dy = xw[1] - x[1]
	return [dx, dy]
def find_eta(x, m, xw1, xw2, k):

	g = -9.8
	h = np.zeros(2)
	d1 = calc_d(x, xw1)
	L1 = x[4]
	if np.linalg.norm(d1) < L1:
	    h[0] = 0
	    h[1] = 0
	else:
	    f1 = k*(np.linalg.norm(d1) - L1)#; %force
	    h[0] = f1*d1[0]/np.linalg.norm(d1)#;
	    h[1] = f1*d1[1]/np.linalg.norm(d1)#;

	return h


def ldm(x, A):
	xt1 = A.dot(x)
	return xt1
	
if __name__== "__main__":
	dtype = torch.float

	N_states = 6
	n_c = 40
	ics = pd.read_csv('./data/ICs2.csv', header=None)
	ics = ics.values
	d_t = pd.read_csv('./data/agg_t.csv', header=None)
	d_t1 = pd.read_csv('./data/agg_t1.csv', header=None)
	data_t = d_t.values
	data_t1 = d_t1.values
	data = {'x': {
		'minus': data_t,
		'plus': data_t1
	}}
	x_minus = data['x'  ]['minus']
	x_plus  = data['x'  ]['plus' ]


	qr = pd.read_csv('./dmdmodels/qr' + str(n_c) + '.csv',header = None)
	A_qr = np.array(qr.values)
	
	model = NeuralNetwork(N_x = 6, N_e = 40)
	checkpoint_stab = torch.load('model.pt')
	model.load_state_dict(checkpoint_stab['model_dict'])

	n_timesteps = 20
	dt = 0.1
	tspan = [0.0, 0.1]
	x_max = 1
	x_min = -1
	vel_max = 20
	vel_min = -20
	base_sse = []
	qr_sse = []
	x_inits = np.array([])
	# calculate stable errors

	index = int(torch.rand(1)*len(ics))
	print(index)
	x_init = ics[index,:]
	x_inits = np.append(x_inits, x_init)
	base_t_e = []
	qr_t_e = []
	g_base = []
	g_qr = []
	X_qrs = []
	X_bases = []
	Xs = []
	X_qrs.append(x_init)
	X_bases.append(x_init)
	Xs.append(x_init)
	xt = torch.tensor(x_init[0:N_states]).type(dtype)
	out = model.g(xt).detach().numpy()
	g_qr.extend(out.tolist())
	x_base = x_init
	x_qr = np.append(x_init, g_qr)
	for p in range(0, n_timesteps):
		sol = solve_ivp(diffeq, tspan, x_init)
		x_init = np.array(sol.y[:,len(sol.y[0]) - 1])
		# print(sol.y)
		# print(x_init)
		Xs.append(x_init)
		xt = torch.tensor(x_base[0:N_states]).type(dtype)
		x_base = model(xt)[0].detach().numpy() #use nn model to generate the prediction
		base_e = np.linalg.norm(x_base[0:N_states] - x_init[0:N_states])
		base_t_e.append(base_e)
		g_base = []
		x_base = x_base[0:N_states]
		X_bases.append(x_base)

		x_qr = ldm(x_qr, A_qr) #use dde layer to predict forward
		qr_e = np.linalg.norm(x_qr[0:N_states] - x_init[0:N_states])
		qr_t_e.append(qr_e)
		g_qr = []
		x_qr = x_qr[0:N_states]
		X_qrs.append(x_qr)
		xt = torch.tensor(x_qr[0:N_states]).type(dtype)
		out = model.g(xt).detach().numpy()
		g_qr.extend(out.tolist())
		x_qr = np.append(x_qr, g_qr)

	# np.savetxt('edmd_traj.csv',X_bases, delimiter = ",")
	# np.savetxt('rdmd_traj.csv',X_qrs, delimiter = ",")
	# np.savetxt('true_traj.csv',Xs, delimiter = ",")

Xs_arr = np.array(Xs)
Xb_arr = np.array(X_bases)
Xqr_arr = np.array(X_qrs)
# print(Xs_arr)
print(n_timesteps)
t = np.linspace(0, n_timesteps, n_timesteps+1)
mdic = {"t" : t, "X_dmd": Xb_arr, "X_dde": Xqr_arr, "X_hist" : Xs_arr}
savemat("est_traj.mat", mdic)
plt.plot(t, Xs_arr[:,0], 'k-', label='Ground Truth')
plt.plot(t, Xb_arr[:,0], 'b-.', label='EDMD')
plt.plot(t, Xqr_arr[:,0], 'g-', label= 'DDE')
plt.legend(fontsize = 'large')
plt.xlabel('Time Steps', fontsize = 'large')
plt.ylabel('X position (meters)', fontsize = 'large')
plt.show()

plt.plot(t, Xs_arr[:,1], 'k-', label='Ground Truth')
plt.plot(t, Xb_arr[:,1], 'b-.', label='EDMD')
plt.plot(t, Xqr_arr[:,1], 'g--', label= 'DDE')
plt.legend(fontsize = 'large')
plt.xlabel('Time Steps', fontsize = 'large')
plt.ylabel('Y position (meters)', fontsize = 'large')

plt.show()

# plt.plot(t, Xs_arr[:,2], 'k-', label='Ground Truth')
# plt.plot(t, Xb_arr[:,2], 'b-.', label='EDMD')
# plt.plot(t, Xqr_arr[:,2], 'g-', label= 'DDE')
# plt.legend(fontsize = 'large')
# plt.xlabel('Time Steps', fontsize = 'large')
# plt.ylabel('X velocity (m/s)', fontsize = 'large')
# plt.show()

# plt.plot(t, Xs_arr[:,3], 'k-', label='Ground Truth')
# plt.plot(t, Xb_arr[:,3], 'b-.', label='EDMD')
# plt.plot(t, Xqr_arr[:,3], 'g-', label= 'DDE')
# plt.legend(fontsize = 'large')
# plt.xlabel('Time Steps', fontsize = 'large')
# plt.ylabel('Y velocity (m/s)', fontsize = 'large')

# plt.show()


# plt.plot(t, Xs_arr[:,4], 'b-', label='Ground Truth')
# plt.plot(t, Xb_arr[:,4], 'r--', label='EDMD')
# plt.plot(t, Xqr_arr[:,4], 'g:', label= 'DDE')
# plt.legend(fontsize = 'large')
# plt.xlabel('Time Steps', fontsize = 'large')
# plt.ylabel('Unstretched Cable Length (m)', fontsize = 'large')
# plt.show()

# plt.plot(t, Xs_arr[:,5], 'b-', label='Ground Truth')
# plt.plot(t, Xb_arr[:,5], 'r--', label='EDMD')
# plt.plot(t, Xqr_arr[:,5], 'g:', label= 'DDE')
# plt.legend(fontsize = 'large')
# plt.xlabel('Time Steps', fontsize = 'large')
# plt.ylabel('Unstretched Cable Velocity (m/s)', fontsize = 'large')

# plt.show()





