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
	model_2 = NeuralNetwork(N_x = 1, N_e = 2)
	checkpoint_2 = torch.load('../model_2.pt')
	model_2.load_state_dict(checkpoint_2['model_dict'])

	model_3 = NeuralNetwork(N_x = 1, N_e = 3)
	checkpoint_3 = torch.load('../model_3.pt')
	model_3.load_state_dict(checkpoint_3['model_dict'])

	model_5 = NeuralNetwork(N_x = 1, N_e = 5)
	checkpoint_5 = torch.load('../model_5.pt')
	model_5.load_state_dict(checkpoint_5['model_dict'])

	model_7 = NeuralNetwork(N_x = 1, N_e = 7)
	checkpoint_7 = torch.load('../model_7.pt')
	model_7.load_state_dict(checkpoint_7['model_dict'])

	model_10 = NeuralNetwork(N_x = 1, N_e = 10)
	checkpoint_10 = torch.load('../model_10.pt')
	model_10.load_state_dict(checkpoint_10['model_dict'])

	val2, vec = model_2.calc_eig()
	val3, vec = model_3.calc_eig()
	val5, vec = model_5.calc_eig()
	val7, vec = model_7.calc_eig()
	val10, vec = model_10.calc_eig()
	  
	# extract real part
	x2 = [ele.real for ele in val2]
	x3 = [ele.real for ele in val3]
	x5 = [ele.real for ele in val5]
	x7 = [ele.real for ele in val7]
	x10 = [ele.real for ele in val10]

	# extract imaginary part
	y2 = [ele.imag for ele in val2]
	y3 = [ele.imag for ele in val3]
	y5 = [ele.imag for ele in val5]
	y7 = [ele.imag for ele in val7]
	y10 = [ele.imag for ele in val10]
		

	figure, axes = plt.subplots( 1 ) 
	# plot the complex numbers
	plt.scatter(x2, y2, label ='2')
	plt.scatter(x3, y3, label ='3')
	plt.scatter(x5, y5, label ='5')
	plt.scatter(x7, y7, label ='7')
	plt.scatter(x10, y10, label ='10')

	angle = np.linspace( 0 , 2 * np.pi , 150 ) 
 
	radius = 1
	 
	x = radius * np.cos( angle ) 
	y = radius * np.sin( angle ) 
	 
	 
	axes.plot( x, y,'--' ) 
	plt.ylabel('Imaginary')
	plt.xlabel('Real')
	leg = plt.legend(loc ='upper right')
	plt.show()