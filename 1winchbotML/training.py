#!/usr/bin/env python
import helpers.learn_modules as lm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


if __name__== "__main__":
    xt = pd.read_csv('./data/agg_t.csv')
    xt1 = pd.read_csv('./data/agg_t1.csv')
    data_t = xt.values
    data_t1 = xt1.values

    # print(data_t)
    # print('###############')
    # print(data_t[0:3])
    # print('##################')
    # print(data_t1.shape)
    # print(data_t.reshape(-1, data_t.shape[-1]))
    data = {'x': {
        'minus': data_t,
        'plus': data_t1
    }}
    # print(data)


    lrn = lm.L3(N_x = 6, N_e = 40)
    lrn.learn(data)
