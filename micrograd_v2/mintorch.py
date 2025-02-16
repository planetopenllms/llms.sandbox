import numpy as np




class Tensor:    
    def __init__(
            self, 
            value, 
            children=(),
            label=None           ## todo/fix - add requires_grad=None - why? why not? 
    ):   
        ## note - ALWAYS use float32  (NOT default is float64 or int64)!!!
        self.value = np.array(value, dtype=np.float32)
        self.grad  = np.zeros_like( self.value )
    
        self._children = children   
        self._label    = label

    def __repr__(self):
        return f"{self.__class__.__name__}{self.value.ndim}D<{self.value.dtype}{self.value.shape}> data={self.value} grad={self.grad}"

    
    def _backward( self, grad ):
         print( " base _backward called; returns None")
         None


    def backward(self):
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = np.ones_like( self.value )

        print( f"==> topo ({len(topo)}):\n", topo )

        for v in reversed(topo):
            print( f"  call backward - {v.__class__.__name__} w/ grad {v.grad.shape} {v.grad}")
            v._backward( v.grad )


  
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

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        print( f"  build __pow__ a {self} ** b {other}")
        return _Pow(self, other)



    def relu(self):
        print( f"  build relu a {self}")
        return _Relu(self)

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




class _Add(Tensor):
    def __init__(self, a, b):
        super().__init__(value=a.value + b.value, children=(a, b))
        self._a = a
        self._b = b

    def _backward(self, grad):
        self._a.grad += grad 
        self._b.grad += grad 
        print( f"  backward _Add grad a {self._a.grad}, b {self._b.grad}")

class _Sub(Tensor):
    def __init__(self, a, b):
        super().__init__(value=a.value - b.value, children=(a, b))
        self._a = a
        self._b = b

    def _backward(self, grad):
        self._a.grad += grad 
        self._b.grad -= grad 
        print( f"  backward _Sub grad a {self._a.grad}, b {self._b.grad}")

      

class _Mul(Tensor):
    def __init__(self, a, b):
        super().__init__(value=a.value * b.value, children=(a, b))
        self._a = a
        self._b = b

    def _backward(self, grad):
        self._a.grad +=  grad * self._b.value 
        self._b.grad +=  self._a.value * grad 
        print( f"  backward _Mul grad a {self._a.grad}, b {self._b.grad}")


class _Pow(Tensor):
    def __init__(self, a, pow):
        super().__init__(value=a.value**pow, children=(a,))
        self._a   = a
        self._pow = pow

    def _backward(self, grad):
        self._a.grad += grad * (self._pow * self._a.value ** (self._pow-1))
        print( f"  backward _Pow**{self._pow} grad a {self._a.grad}")


class _Relu(Tensor):
    def __init__(self, a): 
        super().__init__(value=np.maximum( 0.0, a.value), children=(a,))
        self._a = a

    def _backward(self, grad):
        self._a.grad += grad * (self._a.value > 0.0) 
        print( f"  backward _Relu grad a {self._a.grad}")


class _Tanh(Tensor):
    def __init__(self, a):
        super().__init__(value=np.tanh( a.value), children=(a,))
        self._a = a
    
    def _backward(self, grad):
        self._a.grad += (1 - self.value**2) * grad
        print( f"  backward _Tanh grad a {self._a.grad}")

class _Exp(Tensor):
    def __init__(self, a):
        super().__init__(value=np.exp( a.value ), children=(a,))
        self._a = a

    def _backward(self, grad):  
        self._a.grad += self.value * grad
        print( f"  backward _Exp grad a {self._a.grad}")

class _Log(Tensor):
    def __init__(self, a):
        super().__init__(value=np.log( a.value ), children=(a,))
        self._a = a

    def _backward(self, grad):
        ## or use 1/self._a * grad ???
        self._a.grad +=  grad / self._a.value
        print( f"  backward _Log grad a {self._a.grad}")


