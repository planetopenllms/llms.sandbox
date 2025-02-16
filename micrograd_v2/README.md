
> micrograd is the only piece of code you need to train neural networks. 
> Everything else is just efficiency.


# Micrograd V2

The Micrograd library / code by Andrej Karpathy reworked / extended
to make it work with ndarrays (not just scalars).

See <https://github.com/karpathy/micrograd> for the original 
and <https://github.com/EurekaLabsAI/micrograd> for the official follow-up.


> In this module we build a tiny "autograd" engine (short for automatic gradient) that 
> implements the backpropagation algorithm, as it was prominently popularized 
> for training neural networks in the 1986 paper [Learning Internal Representations by Error Propagation](https://stanford.edu/~jlmcc/papers/PDP/Volume%201/Chap8_PDP86.pdf) by Rumelhart, Hinton and Williams. 
>
> The code we build here is the heart of neural network training - it allows us to calculate
> how we should update the parameters of a neural network in order to make it better at some
> task, such as the one of next token prediction in autoregressive language models. This exact
> same algorithm is used in all modern deep learning libraries, such as PyTorch, TensorFlow,
> JAX, and others, except that those libraries are much more optimized and feature-rich.



What else is different?

- uses numpy.ndarray for Tensor values
  - all Tensors use np.float32 (not the default np.float64 or int64)
- adds a (PyTorch-like) require_grad flag
- adds "typed" tensor classes for operations e.g. _Add, _Mul, _Relu, etc.
  -  "inline" backward closure "unrolled" into a normal method in 
      the "typed" tensor classes

e.g.

``` python
class Value:
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out
```


becomes:

``` python
class Tensor:
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return _Add(self, other)

#### top-level "typed" tensor class
class _Add(Tensor):
    def __init__(self, a, b):
        super().__init__(value=a.value + b.value, children=(a, b))
        self._a = a
        self._b = b

    def _backward(self, grad):
        self._a.grad += grad 
        self._b.grad += grad 
```


or lets see the pow(er) operation; before:

``` python
class Value:
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out
```

becomes

``` python
class Tensor:
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        return _Pow(self, other)

#### top-level "typed" tensor class
class _Pow(Tensor):
    def __init__(self, a, pow):
        super().__init__(value=a.value**pow, children=(a,))
        self._a   = a
        self._pow = pow

    def _backward(self, grad):
        self._a.grad += grad * (self._pow * self._a.value ** (self._pow-1))
```


