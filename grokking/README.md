# Grokking Deep Learning by Andrew Trask




##  A Neural Network in 11 lines of Python

- <https://iamtrask.github.io/2015/07/12/basic-python-network/>
- <https://iamtrask.github.io/2015/07/27/python-network-part2/>
 

``` python
X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
y = np.array([[0,1,1,0]]).T
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1
for j in xrange(60000):
  l1 = 1/(1+np.exp(-(np.dot(X,syn0))))
  l2 = 1/(1+np.exp(-(np.dot(l1,syn1))))
  l2_delta = (y - l2)*(l2*(1-l2))
  l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
  syn1 += l1.T.dot(l2_delta)
  syn0 += X.T.dot(l1_delta)
```

and

``` python
import numpy as np
X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
y = np.array([[0,1,1,0]]).T
alpha,hidden_dim = (0.5,4)
synapse_0 = 2*np.random.random((3,hidden_dim)) - 1
synapse_1 = 2*np.random.random((hidden_dim,1)) - 1
for j in xrange(60000):
    layer_1 = 1/(1+np.exp(-(np.dot(X,synapse_0))))
    layer_2 = 1/(1+np.exp(-(np.dot(layer_1,synapse_1))))
    layer_2_delta = (layer_2 - y)*(layer_2*(1-layer_2))
    layer_1_delta = layer_2_delta.dot(synapse_1.T) * (layer_1 * (1-layer_1))
    synapse_1 -= (alpha * layer_1.T.dot(layer_2_delta))
    synapse_0 -= (alpha * X.T.dot(layer_1_delta))
```



More Articles:
- https://iamtrask.github.io/2014/11/23/harry-potter/
- https://iamtrask.github.io/2015/11/15/anyone-can-code-lstm/
- https://iamtrask.github.io/2015/07/28/dropout/


