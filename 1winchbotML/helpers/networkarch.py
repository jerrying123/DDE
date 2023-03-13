#!/usr/bin/env python

import os
import torch
import math
import copy
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class NeuralNetwork(torch.nn.Module):
    def __init__(self, N_x = 1, N_h = 16, N_e = 10):
        super(NeuralNetwork, self).__init__()
        D_x = N_x
        D_h = N_h
        D_e = N_e
        # Store dimensions in instance variables
        self.D_x = D_x
        self.D_e = D_e
        D_xi = self.D_x + self.D_e

        self.g = nn.Sequential(
            nn.Linear(D_x, D_h),
            nn.ReLU(),
            nn.Linear(D_h, D_h),
            nn.ReLU(),
            nn.Linear(D_h, D_e),
        )
        # Linear dynamic model matrices
        self.A = torch.nn.Linear(D_xi, D_x, bias=False)
        self.H = torch.nn.Linear(D_xi, D_e, bias=False)


    def forward(self, x: torch.Tensor):
        xs   = x
        eta  = self.g(xs)
        xi   = torch.cat((xs,eta))

        x_tp1, eta_tp1 = self.ldm(xi)

        return x_tp1, eta_tp1

    def ldm(self, xi: torch.Tensor):
        return self.A(xi), self.H(xi)

    def calc_eig(self):

        A_ = copy.deepcopy(self.A.weight.data.detach().numpy())
        H_ = copy.deepcopy(self.H.weight.data.detach().numpy())

        arrays = [A_, H_]
        K = np.vstack(arrays)
        w,v = np.linalg.eig(K)
        # print('E-value: ',w)
        # print('E-vector: ', v)

        return w, v

class SuperNetwork(torch.nn.Module):
    def __init__(self, N_x = 1, N_h = 16, N_e = 10):
        super(NeuralNetwork, self).__init__()
        D_x = N_x
        D_h = N_h
        D_e = N_e
        # Store dimensions in instance variables
        self.D_x = D_x
        self.D_e = D_e
        D_xi = self.D_x + self.D_e

        self.g = nn.Sequential(
            nn.Linear(D_x, D_h),
            nn.ReLU(),
            nn.Linear(D_h, D_h),
            nn.ReLU(),
            nn.Linear(D_h, D_e),
        )
        # Linear dynamic model matrices
        self.A = torch.nn.Linear(D_xi, D_x, bias=False)
        self.H = torch.nn.Linear(D_xi, D_e, bias=False)


    def forward(self, x: torch.Tensor):
        xs   = x
        eta  = self.g(xs)
        xi   = torch.cat((xs,eta), 1)

        x_tp1, eta_tp1 = self.ldm(xi)

        return x_tp1, eta_tp1

    def ldm(self, xi: torch.Tensor):
        return self.A(xi), self.H(xi)

    def calc_eig(self):

        A_ = copy.deepcopy(self.A.weight.data.detach().numpy())
        H_ = copy.deepcopy(self.H.weight.data.detach().numpy())

        arrays = [A_, H_]
        K = np.vstack(arrays)
        w,v = np.linalg.eig(K)
        # print('E-value: ',w)
        # print('E-vector: ', v)

        return w, v
