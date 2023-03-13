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
import pickle
import helpers.learn_modules as lm
from helpers.networkarch import NeuralNetwork
import torch
# %% functions

DEBUG = True
VISUALIZE_TREE = False
FailedIDs = []
class DataGraph(object):
	def __init__(self, xt, xtp1, func_size):
		self.xt = xt
		self.xtp1 = xtp1
		self.sorted_lists = []
		self.dim = len(self.xt[0])
		self.func_size = funcs_size
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
		print("Completed graph building")
		n_t = len(triangles.simplices)
		t = 0
		for simplex in triangles.simplices:
			if t%1000 ==0:
				print(str(t) + '/' + str(n_t) + ' triangles computed')
			t += 1
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
					print("Failed on: " + str(simplex))
					global FailedIDs
					FailedIDs.append(simplex)
		print("Completed volume calculations")
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
		self.Q = np.zeros([self.func_size, self.func_size])
		self.R = np.zeros([self.func_size, self.func_size])

		n = len(self.node_list)
		for k in reversed(range(len(self.node_list))):
			"""
			This calculation is for A=QR-1.
			If R is full-rank, then np.linalg.pinv() will calculate np.linalg.inv().
			"""
			if (n-k)%1000 == 0:
				print(f"This is #{(n-k)}/{n}th loop")
			node = self.node_list[k]
			node.calc_RQ()
			self.Q = self.Q + node.Q
			self.R = self.R + node.R
			del self.node_list[k]
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



def identity(x,arg):
	#dummy function to be used as input to tree
	return x
def nn_out(x,model):
	#dummy function to be used as input to tree
	xt = torch.tensor(x[0:N_states]).type(dtype)
	return model.g(xt).detach().numpy()
def dot_product(x, y):
	return np.dot(x, y)

if __name__ =="__main__":
	dtype = torch.float

	N_states = 6
	n_c = 40 #Number of different centers per axis used in creating grid of rbfs
	func_size = N_states + n_c
	model = NeuralNetwork(N_x = 6, N_e = n_c)
	checkpoint_stab = torch.load('model.pt')
	model.load_state_dict(checkpoint_stab['model_dict'])
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
	if os.path.exists('datagraph.obj'):
		filename = open('datagraph.obj','rb')
		graph = pickle.load(filename)
	else:
		filename = open('datagraph.obj','wb')
		graph = DataGraph(x_minus, x_plus)
		pickle.dump(graph, filename)
		if DEBUG:
			# print(FailedIDs)
			print('Number of Failed Nodes: ' + str(len(FailedIDs)))

	if VISUALIZE_TREE:
		tri = graph.triangles
		points = graph.xt
		plt.triplot(points[:,0], points[:,1], tri.simplices)
		# plt.plot(points[:,0], points[:,1], 'o')
		plt.show()
		
	graph.func_size = func_size
	g = []
	g_t1 = []
	func_list = []
	param_list = []
	func_list.append(identity)
	param_list.append([None])

	func_list.append(nn_out)
	param_list.append([model])

	graph.set_funcs(func_list, param_list)
	A = graph.calc_DE()

	# # for node in tree.node_list:
	# # 	print('id :' + str(node.id))
	# # 	print('xt: ' + str(node.xt))
	# # 	print('xtp1: ' + str(node.xtp1))



	qr_str = './dmdmodels/qr' + str(n_c) +'.csv'
	np.savetxt(qr_str, A, delimiter=",")

