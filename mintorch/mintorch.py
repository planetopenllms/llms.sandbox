import numpy as np


class Tensor:    
    def __init__(self, data, _children=(), _op ='', label = '?'):
        self.data = np.array(data)
        self.grad = np.zeros_like( self.data )
    
        self._backward = lambda: None
        self._prev     = set(_children)
        self._op       = _op
        self.label     = label

    ## used by print & co
    def __str__(self):
        return f"Tensor{self.data.ndim}D<{self.data.dtype}{self.data.shape}> data={self.data} grad={self.grad}"

    def __repr__(self):
        return f"Tensor(data={self.data})"
    


    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = np.ones_like( self.data )
        for v in reversed(topo):
            v._backward()


    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        ## note - use += to accumulate gradients (not simply =)!!!
        def _backward():
            self.grad  += out.grad
            other.grad += out.grad
            debug_print( self, out, other )
        
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
            debug_print( self, out, other )
        
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        
        out._backward = _backward
        return out


    def __radd__(self, other): # other + self
        return self + other

    def __rmul__(self, other): # other * self
        return self * other


def debug_print(current, out, other=None):
    if out._op != '' or out._op is not None:
        map_ = {"+": "SumBackward", "*": "MulBackward", "tanh": "TanhBackward"}
        print(f"{map_[out._op]} for {current.label} {'and '+other.label if other else ""}")
    else:
        print(f"Backward for {current.label} {'and '+other.label if other else ""}")
    print("-----------------")
    print(f"Out: {out.label} | Grad:", out.grad)
    print(f"self ({current.label}) | Grad: ", current.grad)
    
    if other is not None:
        print(f"other ({other.label}) | Grad: ", other.grad)

