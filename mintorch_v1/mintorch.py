import numpy as np

##  "global" helpers
##    note - avoid conflict with relu in tensor (add calc_ prefix!!!)

def calc_relu(x):
    return np.maximum( 0, x )

def calc_relu_deriv(x):
    return np.where(x > 0, 1, 0)



class Tensor:    
    def __init__(self, data, _children=(), _op =''):
        ## note - ALWAYS use float32  (NOT default float64 or int64)!!!
        self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like( self.data )
    
        self._backward = lambda: None
        self._prev     = set(_children)
        self._op       = _op

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
        
        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data - other.data, (self, other), '-')

        ## note - use += to accumulate gradients (not simply =)!!!
        def _backward():
            self.grad  += out.grad
            other.grad -= out.grad
        
        out._backward = _backward
        return out


    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad  += other.data * out.grad
            other.grad += self.data * out.grad
        
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

    def __neg__(self): # -self
        return self * -1

    def __rsub__(self, other): # other - self
        return other -self


    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1



 
    def mm(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        ##  use np.matmul or such - why? why not?
        out = Tensor(np.dot(self.data, other.data), (self,other), 'mm')

        def _backward():
             self.grad  += np.dot(out.grad, other.data.T)
             other.grad += np.dot(self.data.T, out.grad)
 
        out._backward = _backward
        return out

    def T(self):
        out = Tensor(self.data.T, (self,), 'T' )

        def _backward():
            self.grad += out.grad.T

        out._backward = _backward
        return out

    def sum(self, axis):  ## make axis required  - was -- axis=None
        out = Tensor(self.data.sum(axis=axis), (self,), 'sum')

        def _backward():
            self.grad += np.ones_like(self.data) * np.expand_dims(out.grad, axis)

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor( calc_relu( self.data ), (self,), 'relu')

        def _backward():
            self.grad += calc_relu_deriv( out.data ) * out.grad
        
        out._backward = _backward
        return out


