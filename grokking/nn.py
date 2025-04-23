###
# based on A Neural Network in 11 lines of Python
#   see https://iamtrask.github.io/2015/07/12/basic-python-network/


##
## 3-layer network (with hidden layer, that is, input/hidden/output)


import numpy as np


def sigmoid(x):
	return 1/(1+np.exp(-x))

def sigmoid_deriv(x):
    return x*(1-x)  



X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])
                
y = np.array([[0],
	  		  [1],
			  [1],
			  [0]])

np.random.seed(42)

# randomly initialize our weights with mean 0
w0 = 2*np.random.random((3,4)) - 1
w1 = 2*np.random.random((4,1)) - 1

print( "w0:", w0.shape, w0, "\nw1:", w1.shape, w1 )


l2 = None


for j in range(60000):
	# Feed forward through layers 0, 1, and 2
    l0 = X    ##  X is "mini-batch" of size 4
    l1 = sigmoid(np.dot(l0,w0))
    l2 = sigmoid(np.dot(l1,w1))

    # how much did we miss the target value?
    l2_error = y - l2
    
    if (j % 10000) == 0:
        print( f"Error: {np.mean(np.abs(l2_error))}" )
        
    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    l2_delta = l2_error * sigmoid_deriv(l2)

    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dot(w1.T)
    
    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l1_delta = l1_error * sigmoid_deriv(l1)

    w1 += l1.T.dot(l2_delta)
    w0 += l0.T.dot(l1_delta)


print( "l2:", l2.shape, l2)
print( "bye" )