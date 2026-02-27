import pandas as pd
import scipy as sp
import numpy as np

def ReLU(x=list[float]):
    if x > 0:
        return x
    else:
        return 0

def forward(w = list[list[float]], prev_nodes = list[float]):
    next_node = np.dot(w,prev_nodes)
    return next_node

def Neural_Net(n_layers = int, n_nodes = list[int]):
    n = np.dot(n_layers, n_nodes)
    return n

#n = Neural_Net(n_layers=2, n_nodes=[2,3])
#print(n)

#curnode = forward(w=[[0.9,0.3],[1,0.6]], prev_nodes=[0.6,0.8])
#print(curnode)