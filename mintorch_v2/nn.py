import numpy as np
from mintorch import Tensor



class SGD():
    def __init__(self, parameters, lr=0.1):
        self.parameters = parameters
        self.lr         = lr
    
    def zero(self):
        for p in self.parameters:
            p.grad *= 0
        
    def step(self, zero=True):
        for p in self.parameters:
            p.value -= p.grad * self.lr
            if(zero):
                p.grad *= 0


class Layer():    
    def __init__(self):
        self.parameters = list()
        
    def get_parameters(self):
        return self.parameters


class Linear(Layer):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        W = np.random.randn(n_inputs, n_outputs) * np.sqrt(2.0/(n_inputs))
        self.weight = Tensor(W)                    # , autograd=True)
        self.bias   = Tensor(np.zeros(n_outputs))  # , autograd=True)
        
        self.parameters.append(self.weight)
        self.parameters.append(self.bias)

    def forward(self, input):
        return input.mm(self.weight)+self.bias.expand(0,len(input.value))



class Sequential(Layer):
    def __init__(self, layers=list()):
        super().__init__()
        
        self.layers = layers
    
    def add(self, layer):
        self.layers.append(layer)
        
    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input
    
    def get_parameters(self):
        params = list()
        for l in self.layers:
            params += l.get_parameters()
        return params
    
