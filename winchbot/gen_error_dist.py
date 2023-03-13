#!/usr/bin/env python
from numpy import exp, sin, cos, log, pi, sign

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import solve_ivp
import helpers.learn_modules as lm
from helpers.networkarch import NeuralNetwork
import torch
# %% functions
def RBF(x, c, epsilon, xind, kinds="gauss"):
	# print('x:' + str(x))
	# print('c:' + str(c))
	# print('e:' + str(epsilon))

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

	model = NeuralNetwork(N_x = 6, N_e = 40)
	checkpoint_stab = torch.load('model.pt')
	model.load_state_dict(checkpoint_stab['model_dict'])


	qr = pd.read_csv('./dmdmodels/qr' + str(n_c) + '.csv',header = None)
	A_qr = np.array(qr.values)



	num_traj = 100
	n_timesteps = 100
	dt = 0.1
	tspan = [0.0, 0.1]
	x_max = 1
	x_min = -1
	vel_max = 20
	vel_min = -20
	base_sse = []
	qr_sse = []
	x_inits = np.array([])
	X_bases = []
	X_qrs = []
	Xs = []
	for m in range(0, num_traj):
		if m % 10 == 0:
			print('Completed ' + str(m) +' / ' + str(num_traj) + ' trajectories')
		# calculate stable errors
		x_init = np.array(ics[m,:])
		x_inits = np.append(x_inits, x_init)
		base_t_e = []
		qr_t_e = []
		g_base = []
		g_qr = []
		xt = torch.tensor(x_init[0:N_states]).type(dtype)
		out = model.g(xt).detach().numpy()
		g_qr.extend(out.tolist())
		x_qr = np.append(x_init, g_qr)
		x_base = x_init
		Xs.append(x_init)
		X_qrs.append(x_qr)
		X_bases.append(x_base)
		for p in range(0, n_timesteps):
			sol = solve_ivp(diffeq, tspan, x_init)
			x_init = np.array(sol.y[:,len(sol.y[0]) - 1])
			# print(sol.y)
			# print(x_init)
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

		qr_sse.append(qr_t_e)
		base_sse.append(base_t_e)

	np.savetxt('edmd_error.csv',base_sse, delimiter = ",")
	np.savetxt('rdmd_error.csv',qr_sse, delimiter = ",")



