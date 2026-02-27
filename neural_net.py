import scipy as sp
import numpy as np
#test
def ReLU(x):
    if not isinstance(x, np.ndarray):
        raise TypeError("exxpected type np.ndarray")
    for i, index in enumerate(x):
        if x[i] < 0:
            x[i] = 0
    return x

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
    # Go through NN once and set weights - 
    # Then use backprop to train the network
    # Then return the trained model with weights + nodes so that only an X can be passed through
    weights = {}
    weights["0"] = np.random.randn(n_nodes[0], len(X))  # np.ones((n_nodes[0], len(X)))
    for i in range(1, n_layers):
        weights[f"{i}"] = np.random.randn(n_nodes[i], n_nodes[i-1])   # np.ones((n_nodes[i], n_nodes[i-1]))
    weights[f"{n_layers}"] = np.random.randn(1, n_nodes[i-1])   # np.ones((1, n_nodes[i-1]))
    layer_nodes = {}
    layer_nodes["l1"] = ReLU(x = weights["0"] @ X)
    for i in range(2, n_layers+1):
        layer_nodes[f"l{i}"] = ReLU(x = weights[f"{i-1}"] @ layer_nodes[f"l{i-1}"])
    
    
    return layer_nodes, n_layers, n_nodes, n_outputs, weights

class Dog:
    def __init__(self, name, breed):
        self.name = name
        self.breed = breed

    def bark(self):
        return f"{self.name} says woof!"
    

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
class NN:
        def __init__(self, layer_nodes, n_layers, n_nodes, n_outputs, weights):
            self.layer_nodes = layer_nodes
            self.n_layers = n_layers
            self.n_nodes = n_nodes
            self.n_outputs = n_outputs
            self.weights = weights


layer_nodes, n_layers, n_nodes, n_outputs, weights = Train_Neural_Net(X = np.array([3,2,1]), y = np.array([1,0,0]), n_outputs = 2, n_layers = 5, n_nodes = np.array([3,3,6,5,4]))
model = NN(layer_nodes=layer_nodes,n_layers=n_layers,n_nodes=n_nodes,n_outputs=n_outputs,weights=weights)

print(model.layer_nodes)