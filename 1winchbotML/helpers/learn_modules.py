#!/usr/bin/env python
import sys
sys.path.append('../')
from abc import ABC, abstractmethod
from collections.abc import Callable
import numpy as np
import itertools, copy, torch, scipy
from scipy import integrate, signal
from enum import Enum
import matplotlib.pyplot as plt
import torch.onnx
import tensorflow as tensorflow
from helpers.networkarch import NeuralNetwork
from onnx_tf.backend import prepare
np.set_printoptions(precision = 4)
np.set_printoptions(suppress = True)

dtype = torch.FloatTensor
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 1
torch.manual_seed(seed)
np.random.seed(seed = seed)
# torch.autograd.set_detect_anomaly(True)
# torch.set_num_threads(8)

DT_DATA_DEFAULT = 0.05
DT_CTRL_DEFAULT = 0.1


class L3():

	def __init__(self, N_x = 1, N_z = 16, N_e = 10, retrain: bool=True,):
		self.n_x = N_x
		self.n_z = N_z
		self.n_e = N_e

		self.retrain = retrain
		self.optimizer = None
		self.model_fn = 'model'
		self.model = None

	def step(self, x_batch: torch.Tensor, y_batch: torch.Tensor, model: torch.nn.Module, loss_fn: Callable):
		# Send data to GPU if applicable
		x_batch = x_batch.to(device)
		y_batch = y_batch.to(device)

		# Parse input
		x_tm1    = x_batch[:,                 :self.n_x         ]

		# Parse output
		x_t      = y_batch[:,                 :self.n_x         ]

		self.input = x_batch
		# Compute "ground truth" eta_t using twin model
		xs_t  = x_t
		eta_t = model.g(xs_t)

		# Propagate x, zeta, and eta using model
		x_hat, eta_hat = model(x_tm1)
		lambda1       = 0.0
		all_linear1_params = torch.cat([x.view(-1) for x in model.g.parameters()])
		l1_regularization = lambda1 * torch.norm(all_linear1_params, 1)
		#create tensor that sums the squares of all parameters corresponding to specific input x or h

		lambda2       = 0.0
		N_x = model.D_x
		N_e = model.D_e
		a = torch.zeros([1,N_x+N_e], dtype = torch.float, device = 'cuda:0', requires_grad=True)
		for param in model.A.parameters():
			for i in range(N_x):
				for j in range(N_x+N_e):
					help_tensor = torch.zeros([1, N_x+N_e], device = 'cuda:0')
					help_tensor[0][j] += torch.square(param[i][j])
					new_tensor = a + help_tensor
					a = new_tensor
					# print(a)

		for param in model.H.parameters():
			for i in range(N_e):
				for j in range(N_x+N_e):
					help_tensor = torch.zeros([1, N_x+N_e], device = 'cuda:0')
					help_tensor[0][j] += torch.square(param[i][j])
					new_tensor = a + help_tensor
					a = new_tensor


		l21_regularization = lambda2 * torch.sum(a)
		# Return
		return loss_fn(x_t, x_hat) + loss_fn(eta_t, eta_hat) + l1_regularization + l21_regularization

	def train_model(self, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, title: str=None):
		model.cuda()
		# Reshape x and y to be vector of tensors
		x = torch.transpose(x,0,1)
		y = torch.transpose(y,0,1)
		# Split dataset into training and validation sets
		N_train = int(3*len(y)/5)
		dataset = torch.utils.data.TensorDataset(x, y)
		train_dataset, val_dataset = torch.utils.data.dataset.random_split(dataset, [N_train,len(y)-N_train])

		# Construct dataloaders for batch processing
		train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32)
		val_loader   = torch.utils.data.DataLoader(dataset=val_dataset  , batch_size=32)

		# Define learning hyperparameters
		loss_fn       = torch.nn.MSELoss(reduction='sum')
		learning_rate = .01
		n_epochs      = 250
		if self.optimizer is None:
			self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
		optimizer     = self.optimizer

		# Initialize arrays for logging
		training_losses = []
		validation_losses = []

		# Main training loop
		try:
			for t in range(n_epochs):
				# Validation
				with torch.no_grad():
					losses = []
					for x, y in val_loader:
						loss = self.step(x, y, model, loss_fn)
						losses.append(loss.item())
					validation_losses.append(np.mean(losses))

				# Terminating condition
				if t>25 and np.mean(validation_losses[-20:-11])<=np.mean(validation_losses[-10:-1]):
					break

				# Training
				losses = []
				for x, y in train_loader:
					loss = self.step(x, y, model, loss_fn)
					losses.append(loss.item())

					optimizer.zero_grad()
					loss.backward()
					optimizer.step()
				training_losses.append(np.mean(losses))

				pstr = f"[{t+1}] Training loss: {training_losses[-1]:.3f}\t Validation loss: {validation_losses[-1]:.3f}"

				print(pstr)

			self.plot_losses(training_losses, validation_losses)
		except KeyboardInterrupt:
			print('Stopping due to keyboard interrupt. Save and continue (Y/N)?')
			ans = input()
			if ans[0].upper() == 'N':
				exit(0)

		model.eval()

		return model

	def plot_losses(self, training_losses, validation_losses, title = None):
		fig, axs = plt.subplots(1,1)
		axs.semilogy(range(len(  training_losses)),   training_losses, label=  'Training Loss')
		axs.semilogy(range(len(validation_losses)), validation_losses, label='Validation Loss')
		axs.set_xlabel('Epoch')
		axs.set_ylabel('Loss')
		axs.legend()
		if title is not None:
			axs.title(title)    

		plt.show()



	def learn(self, data: dict):
		# Copy data for manipulation
		data = copy.deepcopy(data)

		# Format data
		x_minus = torch.transpose(torch.from_numpy(data['x'  ]['minus']).type(dtype), 0,1)
		x_plus  = torch.transpose(torch.from_numpy(data['x'  ]['plus' ]).type(dtype), 0,1)

		# Initialize model
		if self.model is None:
			self.model = NeuralNetwork(N_x = self.n_x, N_e = self.n_e, N_h = self.n_z)
		x_minus.to('cuda:0')
		x_plus.to('cuda:0')
		# Train/load model
		if self.retrain:
			self.model = self.train_model(self.model, x_minus, x_plus).to('cuda:0')
			torch.save({'model_dict': self.model.state_dict()},'{}.pt'.format(self.model_fn))
			torch.save({'opt_dict': self.optimizer.state_dict()},'optimizer.pt')


			# x = torch.randn(batch_size, 2, 224, 224, requires_grad=True)

			#converts model to onnx
			torch.onnx.export(self.model,               # model being run
				  self.input,                         # model input (or a tuple for multiple inputs)
				  "model.onnx",   # where to save the model (can be a file or file-like object)
				  export_params=True,        # store the trained parameter weights inside the model file
				  opset_version=10,          # the ONNX version to export the model to
				  do_constant_folding=True,  # whether to execute constant folding for optimization
				  input_names = ['input'],   # the model's input names
				  output_names = ['output'], # the model's output names
				  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
								'output' : {0 : 'batch_size'}})
			#converts model to keras
			
		else:
			self.model.load_state_dict(torch.load('{}.pt'.format(self.model_fn)))

			self.trained = True

	def augmented_state(x):
		x_shape = x.shape
		if len(x_shape)==3:
			x = x.reshape(-1, x.shape[-1])

		x = torch.from_numpy(x.T).type(dtype)

		
		xs = x
		
		eta = self.model.g(xs)

		xs = torch.cat((xs,eta), 0)
		self.augmented_state = augmented_state

		return xs.detach().numpy()

	def regress_new_LDM(self, data):
		# Copy data for manipulation
		data = copy.deepcopy(data)

		p_data = self.generate_data_ldm(data)

		# Format data
		x_minus = self.flatten_trajectory_data(data['x'  ]['minus'])
		z_minus = self.flatten_trajectory_data(data['eta']['minus'])
		x_plus  = self.flatten_trajectory_data(data['x'  ]['plus' ])
		z_plus  = self.flatten_trajectory_data(data['eta']['plus' ])

		xs_minus = []
		xs_plus = []
		for i in range(len(x_minus)):
			xa_minus = self.augmented_state(x_minus[i])
			xa_plus  = self.augmented_state(x_plus [i])

			xs_minus.append(np.concatenate((xa_minus),0))
			xs_plus.append(xa_plus)

		xs_minus = np.asarray(xs_minus)
		xs_plus  = np.asarray(xs_plus )

		ldm = np.linalg.lstsq(xs_minus,xs_plus,rcond=None)[0].T

		self.model.A.weight.data = torch.from_numpy(ldm[                 :self.n_x         ,:]).type(dtype)
		self.model.H.weight.data = torch.from_numpy(ldm[self.n_x,        :                 ,:]).type(dtype)

		return ldm


	def calc_eig(self):
		A_ = copy.deepcopy(self.model.A.weight.data.detach().numpy())
		H_ = copy.deepcopy(self.model.H.weight.data.detach().numpy())

		K = np.array([[A_],
					[H_]])

		w,v = eig(K)
		print('E-value: ',w)
		print('E-vector: ', v)

		return w, v


	# def generate_data_ldm(self, data):
	# 	self.model.


	def flatten_trajectory_data(data: np.ndarray):
		return data.reshape(-1, data.shape[-1])