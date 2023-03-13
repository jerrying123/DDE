#!/usr/bin/env python
import sys
sys.path.append('../')
import os
import torch
import math
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import helpers.learn_modules as lm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from helpers.networkarch import NeuralNetwork
from onnx_tf.backend import prepare


if __name__ == '__main__':
	model_1 = NeuralNetwork(N_x = 1, N_e = 10)
	checkpoint_1 = torch.load('./../model_stab.pt')
	model_1.load_state_dict(checkpoint_1['model_dict'])

	model_2 = NeuralNetwork(N_x = 1, N_e = 10)
	checkpoint_2 = torch.load('./../model_unst.pt')
	model_2.load_state_dict(checkpoint_2['model_dict'])

	model_3 = NeuralNetwork(N_x = 1, N_e = 20)
	checkpoint_3 = torch.load('./../model_agg.pt')
	model_3.load_state_dict(checkpoint_3['model_dict'])

	n_x = model_1.D_x
	n_e = model_1.D_e
	x_inc = 1
	xdot_inc = 0.1
	x_in = [-10.0]
	x_tmp = [0.0]

	xs = np.zeros([1])
	gs1 = np.zeros([1,n_e])
	gs2 = np.zeros([1,n_e])
	gs3 = np.zeros([1,n_e*2])

	for i in range(100):
		x_tmp[0] = x_in[0] + x_inc * i
		x = torch.tensor(x_tmp)

		xs = np.append(xs, x.detach().numpy(),0)
		out = model_1.g(x)
		# print(np.array([out.detach().numpy()]))
		gs1 = np.append(gs1, np.array([out.detach().numpy()]),0)

		out = model_2.g(x)
		gs2 = np.append(gs2, np.array([out.detach().numpy()]),0)

		out = model_3.g(x)
		gs3 = np.append(gs3, np.array([out.detach().numpy()]),0)

	xs = np.delete(xs, 0, 0)
	gs1 = np.delete(gs1, 0, 0)
	gs2 = np.delete(gs2, 0, 0)
	gs3 = np.delete(gs3, 0, 0)

	fig = plt.figure()
	ax = plt.axes()
	for i in range(n_e):
		ax.scatter(xs, gs1[:,i])
	plt.show()
	# print(xs)
	# print(gs)
	# print(model.g(x))