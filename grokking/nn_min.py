##
## 3-layer network (with hidden layer, that is, input/hidden/output)


import numpy as np


def sigmoid(x):
	return 1/(1+np.exp(-x))

def sigmoid_deriv(x):
    return x*(1-x)  


X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
y = np.array([[0],[1],[1],[0]])

w0 = 2*np.random.random((3,4)) - 1
w1 = 2*np.random.random((4,1)) - 1
print( "w0:", w0.shape, w0, "\nw1:", w1.shape, w1 )


l2 = None
for j in range(60000):
    l0 = X
    l1 = sigmoid(np.dot(X,w0))
    l2 = sigmoid(np.dot(l1,w1))
    
    l2_error = y - l2
    l2_delta = l2_error * sigmoid_deriv(l2)
    
    l1_error = l2_delta.dot(w1.T)
    l1_delta = l1_error * sigmoid_deriv(l1)
    
    w1 += l1.T.dot(l2_delta)
    w0 += l0.T.dot(l1_delta)

print( "l2:", l2.shape, l2)
print( "bye" )

