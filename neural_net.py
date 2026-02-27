import scipy as sp
import numpy as np
#test
def Train_Neural_Net(X, y, n_outputs, n_layers, n_nodes):
    if not isinstance(X, np.ndarray):
        raise TypeError("expected type np.ndarray")
    if not isinstance(y, np.ndarray):
        raise TypeError("expected type np.ndarray")
    if not isinstance(n_outputs, int):
        raise TypeError("expected type integer")
    if not isinstance(n_layers, int):
        raise TypeError("expected type integer")
    if not isinstance(n_nodes, np.ndarray):
        raise TypeError("expected type np.ndarray")
    if n_layers != len(n_nodes):
        raise IndexError(f"number of layers ({n_layers}) given by variable n_layers is different to the number of layers given ({len(n_nodes)}) by n_nodes")
    # Go through NN once and set weights
    # Then use backprop to train the network
    # Then return the trained model with weights + nodes so that only an X can be passed thorugh
    
    weights = {}
    weights["0"] = np.ones((n_nodes[0], len(X)))
    for i in range(1, n_layers):
        # Store in a dictionary with a tuple shape
        weights[f"{i}"] = np.ones((n_nodes[i], n_nodes[i-1]))
    weights[f"{n_layers}"] = np.ones((1, n_nodes[i-1]))
    l1n = weights["0"] @ X
    for element in weights:
        print(element)
    return l1n, n_layers, n_nodes, n_outputs, weights


def complete():
    # Placeholder
    return 0

def back_prop():
    # Placeholder
    return 0



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
    next_node = w @ prev_nodes
    return next_node

def final_layer(w, prev_nodes):
    if not isinstance(w, np.ndarray):
        raise TypeError("expected type np.ndarray")
    if not isinstance(prev_nodes, np.ndarray):
        raise TypeError("expected type np.ndarray")
    last_node = w @ prev_nodes
    return last_node
#n = Neural_Net(n_layers=2, n_nodes=np.array([2,3]))
#print(n)

#curnode = forward(w=np.array([[6,1],[1,6]]), prev_nodes=np.array([1,4]))
#print(curnode)

#response = ReLU(x = np.array([-0.2,0.5,0.3,-1.2]))
#print(response)

sdft = Train_Neural_Net(X = np.array([3,2,1]), y = np.array([1,0,0]), n_outputs = 2, n_layers = 2, n_nodes = np.array([3,3]))
print(sdft)