import numpy as np



##
## todo - for compat with PyTorch change value back to data - why? why not?
##             check others - who is using value or data or ___??

class Tensor:    
    def __init__(
            self, 
            value, 
            children=(),
            requires_grad=False,
            label=None           ## todo/fix - add requires_grad=None - why? why not? 
    ):   
        ## note - ALWAYS use float32  (NOT default is float64 or int64)!!!
        self.value         = np.array(value, dtype=np.float32)
        self.grad          = np.zeros_like( self.value )
        self.requires_grad = requires_grad
    
        self._children = children   
        self._label    = label

    def __repr__(self):
        return f"{self.__class__.__name__}{self.value.ndim}D<{self.value.dtype}{self.value.shape}> data={self.value} grad={self.grad}"

    
    def _backward( self ):
         print( " base _backward called; returns None")
         None


    def backward(self):
        ## assert self.value.ndim == 0, f"backward can only be called for scalar tensors, but it has shape {self.value.shape})"

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            ## todo/check - add v._children too for requires grad check? - why? why not?
            if v.requires_grad and v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        # fill in the first grad with one.  
        self.grad = np.ones_like( self.value )

        print( f"==> topo ({len(topo)}):\n", topo )

        for v in reversed(topo):
            ## assert (v.grad is not None)
            print( f"  call backward - {v.__class__.__name__} w/ grad {v.grad.shape} {v.grad}")
            v._backward()


  
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        print( f"  build __add__ a {self} + b {other}")
        return _Add(self, other)

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        print( f"  build __sub__ a {self} - b {other}")
        return _Sub(self, other)


    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        print( f"  build __mul__ a {self} * b {other}")
        return _Mul(self, other)

    def __pow__(self, pow):
        assert isinstance(pow, (int, float)), "only supporting int/float powers for now"
        print( f"  build __pow__ a {self} ** {pow}")
        return _Pow(self, pow)



    def relu(self):
        print( f"  build relu a {self}")
        return _Relu(self)

    def sigmoid(self):
        print( f"  build sigmoid a {self}" )
        return _Sigmoid(self)


    def tanh(self):
        print( f"  build tanh a {self}")
        return _Tanh(self)

    def exp(self):
        print( f"  build exp a {self}")
        return _Exp(self)

    def log(self):
        print( f"  build (natural) log a {self}")
        return _Log(self)



    def __neg__(self): # -self            -- turn neg(ation) in mul(tiplication) by -1
        return self * -1

    def __truediv__(self, other): # self / other
        return self * other**-1

    ## handle reverse operations
    ##    todo - check order - matrix operations may not be associative!!!
    ##    always use other  OP self   to keep order ????
    ##    check to make it work - other must come first - why? why not?
    ##  check why other + self leads to recursion
    ##
    ##  note - assumes left side (other) is NOT a tensor type
    ##            MUST convert to tensor before call __add/mul/etc.__ operation
    def __radd__(self, other): # other + self   -- reverse add
        return Tensor(other) + self 

    def __rmul__(self, other): # other * self   -- reverse mul(tiplication)
        return Tensor(other) * self 

    def __rsub__(self, other): # other - self
        return Tensor(other) - self 

    def __rtruediv__(self, other): # other / self
        return Tensor(other) / self  

    ##########
    ## more operations
    def mm(self, other):
        ##  note -  use __matmul__ now built-into python (via @) - why? why not?
        ##    keep mm as alias?
        print( f"  build mm a {self} @ b {other}")
        return _MatMul( self, other )
 
    def sum(self, axis):
        assert isinstance(axis, int), "only supporting int for axis for now"
        print( f"  build __sum__ a {self} axis={axis}")
        return _Sum( self, axis )


    def expand(self, dim, copies):
        assert isinstance(dim, int),    "only supporting int for dim for now"
        assert isinstance(copies, int), "only supporting int for copies for now"
        ## change dim to axis - why? why not?
        return _Expand( self, dim, copies )      
  

class _Add(Tensor):
    def __init__(self, a, b):
        super().__init__(value=a.value + b.value, 
                         children=(a, b), requires_grad=a.requires_grad or b.requires_grad )
        self._a = a
        self._b = b

    def _backward(self):
        self._a.grad += self.grad 
        self._b.grad += self.grad 
        print( f"  backward _Add grad a {self._a.grad}, b {self._b.grad}")

class _Sub(Tensor):
    def __init__(self, a, b):
        super().__init__(value=a.value - b.value, 
                         children=(a, b), requires_grad=a.requires_grad or b.requires_grad )
        self._a = a
        self._b = b

    def _backward(self):
        self._a.grad += self.grad 
        self._b.grad -= self.grad 
        print( f"  backward _Sub grad a {self._a.grad}, b {self._b.grad}")

      

class _Mul(Tensor):
    def __init__(self, a, b):
        super().__init__(value=a.value * b.value, 
                         children=(a, b), requires_grad=a.requires_grad or b.requires_grad)
        self._a = a
        self._b = b

    def _backward(self):
        self._a.grad +=  self.grad * self._b.value 
        self._b.grad +=  self._a.value * self.grad 
        print( f"  backward _Mul grad a {self._a.grad}, b {self._b.grad}")


class _Pow(Tensor):
    def __init__(self, a, pow):
        ## note - use np.power( a.value, pow ) - why? why not?
        super().__init__(value=a.value ** pow, 
                         children=(a,), requires_grad=a.requires_grad)
        self._a   = a
        self._pow = pow

    def _backward(self):
        self._a.grad += self.grad * (self._pow * self._a.value ** (self._pow-1))
        print( f"  backward _Pow**{self._pow} grad a {self._a.grad}")


class _Relu(Tensor):
    def __init__(self, a): 
        super().__init__(value=np.maximum( 0.0, a.value), 
                         children=(a,), requires_grad=a.requires_grad)
        self._a = a

    def _backward(self):
        self._a.grad += self.grad * (self._a.value > 0.0) 
        print( f"  backward _Relu grad a {self._a.grad}")


class _Sigmoid(Tensor):
    def __init__(self, a): 
        super().__init__(value=(1 / (1 + np.exp(-a.value))), 
                         children=(a,), requires_grad=a.requires_grad)
        self._a = a

    def _backward(self):
        ## todo/check formula for deriv 
        ones = np.ones_like( self.grad )
        self._a.grad +=  self.grad * (self.value * (ones - self.value )) 
        print( f"  backward _Sigmoid grad a {self._a.grad}")

  

class _Tanh(Tensor):
    def __init__(self, a):
        super().__init__(value=np.tanh( a.value), 
                         children=(a,), requires_grad=a.requires_grad)
        self._a = a
    
    def _backward(self):
        ## notes - use self.value*self.value   for self.value**2
        ##         use simply 1   for   ones = np.ones_like( self.grad )  ???
        ones = np.ones_like( self.grad )
        self._a.grad += (ones - self.value**2) * self.grad
        print( f"  backward _Tanh grad a {self._a.grad}")

class _Exp(Tensor):
    def __init__(self, a):
        super().__init__(value=np.exp( a.value ), 
                         children=(a,), requires_grad=a.requires_grad)
        self._a = a

    def _backward(self):  
        self._a.grad += self.value * self.grad
        print( f"  backward _Exp grad a {self._a.grad}")

class _Log(Tensor):
    def __init__(self, a):
        super().__init__(value=np.log( a.value ), 
                         children=(a,), requires_grad=a.requires_grad)
        self._a = a

    def _backward(self):
        ## or use 1/self._a * grad ???
        self._a.grad +=  self.grad / self._a.value
        print( f"  backward _Log grad a {self._a.grad}")


class _MatMul(Tensor):
    def __init__(self, a, b): 
        assert a.value.ndim == b.value.ndim
        super().__init__(value=np.matmul( a.value, b.value), 
                         children=(a,b), requires_grad=a.requires_grad or b.requires_grad) 
        self._a = a
        self._b = b

    def _backward(self):
        self._a.grad +=  np.matmul( self.grad, self._b.value.T )
        self._b.grad +=  np.matmul( self._a.value.T, self.grad )
        print( f"  backward _MatMul grad a {self._a.grad}, b {self._b.grad}")


class _Sum(Tensor):
    def __init__(self,a,axis):
        super().__init__(value=a.value.sum(axis=axis), 
                         children=(a,), requires_grad=a.requires_grad)
        self._a    = a
        self._axis = axis

    def _backward(self):
        ##  todo - check if grad formula is working/good - why? why not? 
        ones = np.ones_like(self.value)
        self._a.grad += ones * np.expand_dims(self.grad, axis=self._axis)
        print( f"  backward _Sum(axis={self._axis}) grad a {self._a.grad}")

## use?  why? why not?  from grokking deep learning
#                   if("sum" in self.creation_op):
#                     dim = int(self.creation_op.split("_")[1])
#                     self.creators[0].backward(self.grad.expand(dim,
#                        self.creators[0].data.shape[dim]))
   
class _Expand(Tensor):
    def __init__(self, a, dim, copies):
        trans_cmd = list(range(0,len(a.value.shape)))
        trans_cmd.insert(dim,len(a.value.shape))
        new_value = a.value.repeat(copies).reshape(list(a.value.shape) + [copies]).transpose(trans_cmd)

        super().__init__(value=new_value, 
                         children=(a,), requires_grad=a.requires_grad)
        self._a      = a
        self._dim    = dim
        self._copies = copies

    def _backward(self):
        self._a.grad += self.grad.sum(axis=self._dim)
        print( f"  backward _Expand(dim={self._dim}) grad a {self._a.grad}")
                 