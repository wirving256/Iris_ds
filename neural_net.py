import pandas as pd
import scipy as sp
import numpy as np

def ReLU(x):
    if not isinstance(x, np.ndarray):
        raise TypeError("exxpected type np.ndarray")
    for i, index in enumerate(x):
        if x[i] < 0:
            x[i] = 0

    return x

def forward(w, prev_nodes):
    if not isinstance(w, np.ndarray):
        raise TypeError("expected type np.ndarray")
    if not isinstance(prev_nodes, np.ndarray):
        raise TypeError("expected type np.ndarray")
    next_node = np.dot(w,prev_nodes)
    return next_node

def Neural_Net(n_layers, n_nodes):
    if not isinstance(n_layers, int):
        raise TypeError("expected type integer")
    if not isinstance(n_nodes, np.ndarray):
        raise TypeError("expected type np.ndarray")
    if n_layers != len(n_nodes):
        raise IndexError(f"number of layers ({n_layers}) given by variable n_layers is different to the number of layers given ({len(n_nodes)}) by n_nodes")
    return n_layers

#n = Neural_Net(n_layers=2, n_nodes=np.array([2,3]))
#print(n)

#curnode = forward(w=np.array([[6,1],[1,6]]), prev_nodes=np.array([1,4]))
#print(curnode)

#response = ReLU(x = np.array([-0.2,0.5,0.3,-1.2]))
#print(response)