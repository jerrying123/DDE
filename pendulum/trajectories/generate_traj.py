#!/usr/bin/env python
from numpy import exp, sin, cos, log, pi, sign

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import solve_ivp
from scipy.io import loadmat
import random
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

def diffeq(t, y):
	# k = 200; %stiffness of the wall
	# c = 1; %linear damping coefficient
	# dx1 = x(2);
	# Fk = 0;
	# if abs(x(1)) > pi/4
	#     Fk = -sign(x(1))*k*(abs(x(1)) - pi/4)^2;
	# end
	# dx2 = -sin(x(1)) + Fk - sign(x(2))*c*x(2)^2; %g/l is set to 1

	# dxdt = [dx1; dx2];
	k = 200
	c = 1
	Fk = 0
	if np.abs(y[0]) > pi/4:
		Fk = -sign(y[0]) * k * (np.abs(y[0]) - pi/4)**2

	dx1 = y[1]
	dx2 = -sin(y[0]) + Fk - sign(y[1])*c*y[1]**2
	return [dx1, dx2]


def ldm(x, A):
	xt1 = A.dot(x)
	return xt1
	
if __name__== "__main__":
	N_states = 2
	n_c = 9
	datasize = 5000
	d_t = pd.read_csv('./data/agg_t_' +str(datasize) + '.csv', header=None)
	d_t1 = pd.read_csv('./data/agg_t1_' +str(datasize) + '.csv', header=None)
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
	zeros_vec = np.zeros(rbf_c1.shape)
	rbf_c = np.array(np.meshgrid(rbf_c1, rbf_c2)).T.reshape(-1, N_states)

	rbf_dilation = np.ones(rbf_c.shape,dtype="float64")


	qr = pd.read_csv('./models/qr' + str(n_c) + '_' +str(datasize) + '.csv',header = None)
	A_qr = np.array(qr.values)

	base = pd.read_csv('./models/base' + str(n_c) + '_' +str(datasize) + '.csv',header = None)
	A_base = np.array(base.values)

	n_timesteps = 200
	dt = 0.1
	tspan = [0.0, 0.1]
	x_max = 1
	x_min = -1
	vel_max = 20
	vel_min = -20
	base_sse = []
	qr_sse = []
	x_inits = np.array([])
	traj_ics = loadmat('data/ICs2.mat')
	# calculate stable errors
	r1 = np.random.uniform(low = x_min, high = x_max)
	r2 = np.random.uniform(low = vel_min, high = vel_max)
	print(len(traj_ics['X']))
	traj_num = random.randint(0,len(traj_ics['X']))
	x_init = traj_ics['X'][traj_num]#np.array([r1, r2])
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
	for j in range(0, len(rbf_c)):
		g_base.append(RBF(x_init, rbf_c[j,:], rbf_dilation[j,0]))
		g_qr.append(RBF(x_init, rbf_c[j,:], rbf_dilation[j,0]))
	x_base = np.append(x_init, g_base)
	x_qr = np.append(x_init, g_qr)
	for k in range(0, n_timesteps):
		sol = solve_ivp(diffeq, tspan, x_init)
		x_init = np.array(sol.y[:,len(sol.y[0]) - 1])
		# print(sol.y)
		# print(x_init)
		Xs.append(x_init)
		x_base = ldm(x_base, A_base)
		base_e = np.linalg.norm(x_base[0:N_states] - x_init[0:N_states])
		base_t_e.append(base_e)
		g_base = []
		x_base = x_base[0:N_states]
		X_bases.append(x_base)
		for j in range(0, len(rbf_c)):
			g_base.append(RBF(x_base, rbf_c[j,:], rbf_dilation[j,0]))
		x_base = np.append(x_base, g_base)

		x_qr = ldm(x_qr, A_qr)
		qr_e = np.linalg.norm(x_qr[0:N_states] - x_init[0:N_states])
		qr_t_e.append(qr_e)
		g_qr = []
		x_qr = x_qr[0:N_states]
		X_qrs.append(x_qr)
		for j in range(0, len(rbf_c)):
			g_qr.append(RBF(x_qr, rbf_c[j,:], rbf_dilation[j,0]))
		x_qr = np.append(x_qr, g_qr)

	# np.savetxt('edmd_traj.csv',X_bases, delimiter = ",")
	# np.savetxt('rdmd_traj.csv',X_qrs, delimiter = ",")
	# np.savetxt('true_traj.csv',Xs, delimiter = ",")

Xs_arr = np.array(Xs)
Xb_arr = np.array(X_bases)
Xqr_arr = np.array(X_qrs)
# print(Xs_arr)
plt.plot(Xs_arr[:,0], Xs_arr[:,1], 'k-', label='Ground Truth')
plt.plot(Xb_arr[:,0], Xb_arr[:,1], 'b-.', label='EDMD')
plt.plot(Xqr_arr[:,0], Xqr_arr[:,1], 'g--', label= 'DDE')
plt.legend(fontsize = 'large')
plt.show()


t = np.linspace(0, n_timesteps, n_timesteps+1)

plt.plot(t, Xs_arr[:,0], 'k-', label='Ground Truth')
plt.plot(t, Xb_arr[:,0], 'b-.', label='EDMD')
plt.plot(t, Xqr_arr[:,0], 'g--', label= 'DDE')
plt.legend(fontsize = 'large')
plt.xlabel('Time Steps', fontsize = 'large')
plt.ylabel('Angle (radians)', fontsize = 'large')
plt.show()

plt.plot(t, Xs_arr[:,1], 'k-', label='Ground Truth')
plt.plot(t, Xb_arr[:,1], 'b-.', label='EDMD')
plt.plot(t, Xqr_arr[:,1], 'g--', label= 'DDE')
plt.legend(fontsize = 'large')
plt.xlabel('Time Steps', fontsize = 'large')
plt.ylabel('Angular Velocity (radians/s)', fontsize = 'large')

plt.show()


