import numpy as np
from mintorch import Tensor
import torch



##
## reuse test from micrograd by Andrej Karpathy
##   see https://github.com/karpathy/micrograd/blob/master/test/test_engine.py

def test_grads():
    x = Tensor(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmt, ymt = x, y

    x = torch.Tensor([-4.0]) 
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass went well
    assert ymt.value == ypt.data.item()
    # backward pass went well
    assert xmt.grad == xpt.grad.item()



def test_more_grads():

    a = Tensor(-4.0)
    b = Tensor(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amt, bmt, gmt = a, b, g

    a = torch.Tensor([-4.0]) 
    b = torch.Tensor([2.0])  
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(gmt.value - gpt.data.item()) < tol
    # backward pass went well
    assert abs(amt.grad - apt.grad.item()) < tol
    assert abs(bmt.grad - bpt.grad.item()) < tol



def test_more_grads_cont():

    a = Tensor(-4.0)
    b = Tensor(2.0)
    c = a + b
    d = a.tanh()
    e = Tensor(1.0).log()
    f = (e-c).exp()
    f.backward()
    amt, bmt, fmt = a, b, f

    a = torch.Tensor([-4.0]) 
    b = torch.Tensor([2.0])   
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a.tanh()
    e = torch.Tensor([1.0]).log()
    f = (e-c).exp()
    f.backward()
    apt, bpt, fpt = a, b, f

    tol = 1e-6
    # forward pass went well
    assert abs(fmt.value - fpt.data.item()) < tol
    # backward pass went well
    assert abs(amt.grad - apt.grad.item()) < tol
    assert abs(bmt.grad - bpt.grad.item()) < tol
