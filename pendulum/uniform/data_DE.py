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
import csv
# %% functions

DEBUG = True
VISUALIZE_TREE =False
FailedIDs = []
class DataGraph(object):
	def __init__(self, xt, xtp1):
		self.xt = xt
		self.xtp1 = xtp1
		self.sorted_lists = []
		self.dim = len(self.xt[0])
		self.node_list, self.triangles = self.create_graph()

	def create_graph(self):
		#creates nodes and connects them
		node_nums = np.array([np.linspace(0, len(self.xt)-1, len(self.xt), dtype = int)]).transpose()
		#Create node list
		node_list = [] #list containing all nodes
		for i in range(0, len(self.xt)):
			node_list.append(Node(xt = self.xt[i], xtp1 = self.xtp1[i], ind = i))

		#Create graph using Delaunay Triangles
		triangles = Delaunay(self.xt)
		for simplex in triangles.simplices:
			try:
				points = []
				# print('Simplex: ' + str(*simplex))
				points.extend(self.xt[simplex])
				# print('Points: ' + str(points))
				ch = ConvexHull(points)
				for i in range(0, len(simplex)):
					#assign volume to nodes based on volumes
					node_list[simplex[i]].vol = node_list[simplex[i]].vol + ch.volume/len(simplex)
			except:
				if DEBUG:
					global FailedIDs
					FailedIDs.append(simplex)

		print("Completed graph building")
		return node_list, triangles

	def connect_up(self, node1, node2, dim):
		#node 2 is the closest neighbor along dimension dim above node 1
		node1.connect_up(node2, dim)
		node2.connect_down(node1, dim)

	def set_funcs(self, funcs, params):
		self.func_list = funcs
		self.params = params
		for node in self.node_list:
			node.set_funcs(funcs, params)


	def calc_DE(self):
		#set Q and R sizes
		self.Q = np.zeros([len(self.params)+self.dim-1, len(self.params)+self.dim-1])
		self.R = np.zeros([len(self.params)+self.dim-1, len(self.params)+self.dim-1])

		n = len(self.node_list)
		min_vol = 100000
		max_vol = 0
		for k in reversed(range(len(self.node_list))):
			"""
			This calculation is for A=QR-1.
			If R is full-rank, then np.linalg.pinv() will calculate np.linalg.inv().
			"""
			if (n-k)%1000 == 0:
				print(f"This is #{(n-k)}/{n}th loop")
			node = self.node_list[k]
			if node.vol < min_vol:
				min_vol = node.vol
			if node.vol > max_vol:
				max_vol = node.vol
			node.calc_RQ()
			self.Q = self.Q + node.Q
			self.R = self.R + node.R
			del self.node_list[k]
		print(max_vol)
		print(min_vol)

		self.A = self.Q@np.linalg.pinv(self.R)
		return self.A


class Node(object):
	"""docstring for Node"""
	def __init__(self, xt, xtp1, ind, usehull=True):
		#xt and xtp1 are lists that are containing the state vector corresponding to the node
		self.id = ind
		self.xt = xt 
		self.xtp1 = xtp1
		# print('xt :' + str(self.xt))
		# print('xtp1 : ' + str(self.xtp1))
		self.vol = 0
		self.Q = []
		self.R = []
		self.dim = len(xt)
		self.connections = []

	def set_funcs(self, funcs, params):
		self.funcs = funcs
		self.params = params
		self.Q = np.zeros([len(self.params)+self.dim-1, len(self.params)+self.dim-1])
		self.R = np.zeros([len(self.params)+self.dim-1, len(self.params)+self.dim-1])



	def calc_RQ(self):
		if self.funcs and self.params:
			g = np.array([])
			gF = np.array([])
			for i in range(0, len(self.params)):
				# print(self.xt)
				# print(self.xtp1)
				# print(*self.params[i])
				g = np.append(g, self.funcs[i](self.xt, *self.params[i]))
				gF = np.append(gF, self.funcs[i](self.xtp1, *self.params[i]))
			
			self.gx = np.matrix(g)
			self.gFx = np.matrix(gF)
			# print(gx)
			# print(gFx)
			# print(self.R.shape)
			self.R =  dot_product(self.gx.getH(), self.gx)* self.vol
			self.Q =  dot_product(self.gFx.getH(), self.gx)* self.vol


		
def RBF(x, c, epsilon, kinds="gauss"):
	# print('x:' + str(x))
	# print('c:' + str(c))
	# print('e:' + str(epsilon))

	r = np.linalg.norm(x - c)
	# print('r:' + str(r))
	if kinds == "gauss":
		return exp(-(epsilon*r)**2)
	elif kinds == "quad":
		return 1/(1+(epsilon*r)**2)
	else:
		pass

def identity(x,arg):
	#dummy function to be used as input to tree
	return x

def dot_product(x, y):
	return np.dot(x, y)

if __name__ =="__main__":

	f = open('q.csv', 'w')
	writer = csv.writer(f, delimiter =',')
	n_c = 5 #Number of different centers per axis used in creating grid of rbfs
	N_states = 2
	datasizes = [100,225,625,900,1225,1600,2500]#,10000,25000]
	for dsize in datasizes:
		print('Modeling for: ', dsize)
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

		graph = DataGraph(x_minus, x_plus)
		if DEBUG:
			# print(FailedIDs)
			print('Number of Failed Nodes: ' + str(len(FailedIDs)))

		if VISUALIZE_TREE:
			tri = graph.triangles
			points = graph.xt
			plt.triplot(points[:,0], points[:,1], tri.simplices)
			# plt.plot(points[:,0], points[:,1], 'o')
			plt.show()
			
		rbf_min = np.ndarray.min(x_minus,0)
		rbf_max = np.ndarray.max(x_minus,0)
		rbf_c1 = np.linspace(rbf_min[0], rbf_max[0], n_c, dtype="float64")
		rbf_c2 = np.linspace(rbf_min[1], rbf_max[1], n_c, dtype="float64")
		rbf_c = np.array(np.meshgrid(rbf_c1, rbf_c2)).T.reshape(-1, N_states)

		rbf_dilation = np.ones(rbf_c.shape,dtype="float64")
		g = []
		g_t1 = []
		func_list = []
		param_list = []
		func_list.append(identity)
		param_list.append([None])
		for j in range(0, len(rbf_c)):
			func_list.append(RBF)
			param = [rbf_c[j,:], rbf_dilation[j,0]]
			param_list.append(param)

		graph.set_funcs(func_list, param_list)
		A = graph.calc_DE()

		# # for node in tree.node_list:
		# # 	print('id :' + str(node.id))
		# # 	print('xt: ' + str(node.xt))
		# # 	print('xtp1: ' + str(node.xtp1))

		qarr = [dsize, graph.Q[0,1], graph.Q[3,4],graph.Q[5,7], graph.Q[13,17]]
		writer.writerow(qarr)

		qr_str = './models/qr' + str(n_c) +'_' + str(dsize)+ '.csv'
		np.savetxt(qr_str, A, delimiter=",")

	f.close()

