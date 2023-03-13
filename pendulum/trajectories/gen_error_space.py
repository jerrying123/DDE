#!/usr/bin/env python
from numpy import exp, sin, cos, log, pi, sign
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import solve_ivp
from scipy.io import loadmat
from scipy.spatial import ConvexHull
# %% functions

def point_in_hull(point, hull, tolerance=1e-12):
	return all(
		(np.dot(eq[:-1], point) + eq[-1] <= tolerance)
		for eq in hull.equations)

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
	n_c = 7
	grid_res = 51
	plotting = True
	datasizes = [1000,1500,2500,5000]#[500,800,1000,1500,2500,5000,10000,25000]
	for dsize in datasizes:

		print("Calculating for: " , dsize)
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
		points = x_minus
		hull = ConvexHull(points)
		rbf_min = np.ndarray.min(x_minus,0)
		rbf_max = np.ndarray.max(x_minus,0)
		rbf_c1 = np.linspace(rbf_min[0], rbf_max[0], n_c, dtype="float64")
		rbf_c2 = np.linspace(rbf_min[1], rbf_max[1], n_c, dtype="float64")
		zeros_vec = np.zeros(rbf_c1.shape)
		rbf_c = np.array(np.meshgrid(rbf_c1, rbf_c2)).T.reshape(-1, N_states)

		rbf_dilation = np.ones(rbf_c.shape,dtype="float64")


		qr = pd.read_csv('./models/qr' + str(n_c) + '_' + str(dsize) + '.csv',header = None)
		A_qr = np.array(qr.values)

		base = pd.read_csv('./models/base' + str(n_c) + '_' + str(dsize) + '.csv',header = None)
		A_base = np.array(base.values)

		ridge = pd.read_csv('./models/ridge' + str(n_c) + '_' + str(dsize) + '.csv',header = None)
		A_rdg = np.array(ridge.values)

		num_traj = 100
		n_timesteps = 200
		dt = 0.1
		tspan = [0.0, 0.1]
		x_max = rbf_max[0]
		x_min = -x_max
		vel_max = rbf_max[1]
		vel_min = -vel_max

		xspace = np.linspace(x_min, x_max, grid_res, dtype = "float64")
		velspace = np.linspace(vel_min, vel_max, grid_res, dtype = "float64")
		xgrid, vgrid = np.meshgrid(xspace, velspace, indexing= 'ij')
		errorgrid_b = np.zeros([grid_res,grid_res])
		errorgrid_qr = np.zeros([grid_res,grid_res])
		errorgrid_b_s = np.zeros([grid_res,grid_res])
		errorgrid_qr_s = np.zeros([grid_res,grid_res])
		errorgrid_rdg = np.zeros([grid_res,grid_res])
		errorgrid_rdg_s = np.zeros([grid_res,grid_res])
		# print(xgrid.shape)
		for i in range(0,grid_res):
			for k in range(0, grid_res):
				x_init = np.array([xgrid[i,k], vgrid[i,k]])

				base_t_e = []
				qr_t_e = []
				rdg_t_e = []

				g_base = []
				g_qr = []
				g_rdg = []
				for j in range(0, len(rbf_c)):
					g_base.append(RBF(x_init, rbf_c[j,:], rbf_dilation[j,0]))
					g_qr.append(RBF(x_init, rbf_c[j,:], rbf_dilation[j,0]))
					g_rdg.append(RBF(x_init, rbf_c[j,:], rbf_dilation[j,0]))

				x_base = np.append(x_init, g_base)
				x_rdg = np.append(x_init, g_rdg)
				x_qr = np.append(x_init, g_qr)

				sol = solve_ivp(diffeq, tspan, x_init)
				x_end = np.array(sol.y[:,len(sol.y[1]) - 1])

				x_base = ldm(x_base, A_base)
				base_e = np.linalg.norm(x_base[0:N_states] - x_end[0:N_states])
				base_t_e.append(base_e)
				g_base = []
				x_base = x_base[0:N_states]
				for j in range(0, len(rbf_c)):
					g_base.append(RBF(x_base, rbf_c[j,:], rbf_dilation[j,0]))
				x_base = np.append(x_base, g_base)

				x_rdg = ldm(x_rdg, A_rdg)
				rdg_e = np.linalg.norm(x_rdg[0:N_states] - x_end[0:N_states])
				rdg_t_e.append(rdg_e)
				g_rdg = []
				x_rdg = x_rdg[0:N_states]
				for j in range(0, len(rbf_c)):
					g_rdg.append(RBF(x_rdg, rbf_c[j,:], rbf_dilation[j,0]))
				x_rdg = np.append(x_rdg, g_rdg)

				x_qr = ldm(x_qr, A_qr)
				qr_e = np.linalg.norm(x_qr[0:N_states] - x_end[0:N_states])
				qr_t_e.append(qr_e)
				g_qr = []
				x_qr = x_qr[0:N_states]
				for j in range(0, len(rbf_c)):
					g_qr.append(RBF(x_qr, rbf_c[j,:], rbf_dilation[j,0]))
				x_qr = np.append(x_qr, g_qr)

				# print(base_e)
				errorgrid_b[i,k] = base_e
				errorgrid_rdg[i,k] = rdg_e
				errorgrid_qr[i,k] = qr_e
				if point_in_hull(x_init, hull):
					errorgrid_b_s[i,k] = base_e
					errorgrid_qr_s[i,k] = qr_e
					errorgrid_rdg_s[i,k] = rdg_e


		cs_b = np.sum(np.sum(errorgrid_b_s))
		cs_qr = np.sum(np.sum(errorgrid_qr_s))
		cs_rdg = np.sum(np.sum(errorgrid_rdg_s))

		print("EDMD Error: ", cs_b)
		print("DDE Error: ", cs_qr)
		print("Rdg Error: ", cs_rdg)


		print("EDMD Max Error: ", errorgrid_b_s.max())
		print("DDE Max Error: ", errorgrid_qr_s.max())
		print("Rdg Max Error: ", errorgrid_rdg_s.max())


		print("EDMD Std: ", np.std(errorgrid_b_s))
		print("DDE Std: ", np.std(errorgrid_qr_s))
		print("Rdg Std: ", np.std(errorgrid_rdg_s))

		if plotting:
			# Heat Maps:

			fig, ax = plt.subplots()
			eb_min, eb_max = -np.abs(errorgrid_b).max(), np.abs(errorgrid_b).max()
			c = ax.pcolormesh(xgrid, vgrid, errorgrid_b, vmin= 0, vmax = 0.15)
			fig.colorbar(c, ax=ax, label='Squared Error')
			for simplex in hull.simplices:
				ax.plot(points[simplex, 0], points[simplex, 1], 'c')
			ax.set(xlabel='Angle',ylabel='Angular Velocity')
			plt.title("Squared Error - EDMD")
			plt.show()

			fig, ax = plt.subplots()
			eqr_min, eqr_max = -np.abs(errorgrid_qr).max(), np.abs(errorgrid_qr).max()
			c = ax.pcolormesh(xgrid, vgrid, errorgrid_qr, vmin= 0, vmax = 0.15)
			fig.colorbar(c, ax=ax, label='Squared Error')
			for simplex in hull.simplices:
				ax.plot(points[simplex, 0], points[simplex, 1], 'c')
			ax.set(xlabel='Angle',ylabel='Angular Velocity')
			plt.title("Squared Error - DDE")
			plt.show()

			fig, ax = plt.subplots()
			erdg_min, erdg_max = -np.abs(errorgrid_rdg).max(), np.abs(errorgrid_rdg).max()
			c = ax.pcolormesh(xgrid, vgrid, errorgrid_rdg, vmin= 0, vmax = 0.15)
			fig.colorbar(c, ax=ax, label='Squared Error')
			for simplex in hull.simplices:
				ax.plot(points[simplex, 0], points[simplex, 1], 'c')
			ax.set(xlabel='Angle',ylabel='Angular Velocity')
			plt.title("Squared Error - Ridge")
			plt.show()
			#3D with contour projections

			# ax = plt.figure().add_subplot(projection='3d')
			# ax.plot_surface(xgrid,vgrid, errorgrid_b, edgecolor='royalblue')
			# ax.contour(xgrid, vgrid, errorgrid_b, zdir='x', offset = 1, cmap='coolwarm')
			# ax.contour(xgrid, vgrid, errorgrid_b, zdir='y', offset = -2.2, cmap='coolwarm')
			# plt.title("Squared Error - EDMD")
			# ax.set(xlabel='Angle',ylabel='Angular Velocity', zlabel='Squared Error')
			# plt.show()

			# ax = plt.figure().add_subplot(projection='3d')
			# ax.plot_surface(xgrid,vgrid, errorgrid_qr, edgecolor='royalblue')
			# ax.contour(xgrid, vgrid, errorgrid_qr, zdir='x', offset = 1, cmap='coolwarm')
			# ax.contour(xgrid, vgrid, errorgrid_qr, zdir='y', offset = -2.2, cmap='coolwarm')
			# plt.title("Squared Error - DDE")

			# ax.set(xlabel='Angle',ylabel='Angular Velocity', zlabel='Squared Error')
			# plt.show()

			# # ax = plt.figure().add_subplot(projection='3d')
			# # ax.plot_surface(xgrid,vgrid, errorgrid_qr, edgecolor='green')
			# # ax.plot_surface(xgrid,vgrid, errorgrid_b, edgecolor='royalblue')
			# # ax.set(xlabel='Angle',ylabel='Angular Velocity', zlabel='Squared Error')
			# # plt.show()

			# #2D cross sections
			# #Theta = 0
			# # print(xgrid[25,0])
			# fig, ax = plt.subplots()
			# bep, = ax.plot(vgrid[25,:], errorgrid_b[25,:])
			# qrp, = ax.plot(vgrid[25,:], errorgrid_qr[25,:])
			# bep.set_label('EDMD')
			# qrp.set_label('DDE')
			# ax.set(xlabel='Angular Velocity', ylabel='Squared Error')
			# ax.legend()
			# plt.show()

			# #Thetadot = 0
			# # print(vgrid[0,25])
			# fig, ax = plt.subplots()
			# bep, = ax.plot(xgrid[:,25], errorgrid_b[:,25])
			# qrp, = ax.plot(xgrid[:,25], errorgrid_qr[:,25])
			# bep.set_label('EDMD')
			# qrp.set_label('DDE')
			# ax.set(xlabel='Angular Velocity', ylabel='Squared Error')
			# ax.legend()
			# plt.show()