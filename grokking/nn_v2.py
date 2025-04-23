###
# based on A Neural Network in 11 lines of Python
#   see https://iamtrask.github.io/2015/07/12/basic-python-network/


##
## 3-layer network (with hidden layer, that is, input/hidden/output)


import numpy as np


def sigmoid(x):
    ## Values between 0 and 1 calculated as 1/(1 + exp(-x))
	return 1/(1+np.exp(-x))

def sigmoid_deriv(x):
    ## Derivative values calculated as sigmoid(x) * (1 - sigmoid(x))
    ##  The derivative is largest at x=0 (0.25) 
    ##   and approaches 0 as x moves away from 0 in either direction.
    return x*(1-x)  


## add/try relu
def relu(x):
    return np.maximum( 0.0, x )

def relu_deriv(x):
    ##  The derivative of the ReLU (Rectified Linear Unit) function is:
    ##     1 for x > 0
    ##     0 for x ≤ 0 
    return np.where(x > 0, 1.0, 0.0) 
    ## return (x > 0).astype(np.float32)



X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])
                
y = np.array([[0],
	  		  [1],
			  [1],
			  [0]])

np.random.seed(42)

# randomly initializ - use he initialization 
w0 = np.random.randn(3,4) * np.sqrt(2.0 / 3)  ## 3 = fan_in
w1 = np.random.randn(4,1) * np.sqrt(2.0 / 4)  ## 4 = fan_in

print( "w0:", w0.shape, w0, "\nw1:", w1.shape, w1 )


lr = 0.1    # Add small learning rate
l2 = None


for j in range(60000):
	# Feed forward through layers 0, 1, and 2
    l0 = X    ##  X is "mini-batch" of size 4
    l1 = relu(np.dot(l0,w0))
    l2 = sigmoid(np.dot(l1,w1))  ## note - keep sigmoid for output layer for binary classification

    # how much did we miss the target value?
    l2_error = y - l2
    
    if (j % 10000) == 0:
        print( "w0:", w0.shape, w0, "\nw1:", w1.shape, w1 )
        print( "l2_error", l2_error.shape, l2_error, "\nl2:", l2.shape, l2 )
        print( f"Error: {np.mean(np.abs(l2_error))}" )
        
    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    l2_delta = l2_error * sigmoid_deriv(l2)

    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = l2_delta.dot(w1.T)
    
    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l1_delta = l1_error * relu_deriv(l1)

    # if (j % 1000) == 0:
    #    print( "l2_error", l2_error.shape, l2_error, "\nl2:", l2.shape, l2 )
    #    print( "l2_delta:", l2_delta.shape, l2_delta, "\nl1_delta:", l1_delta.shape, l1_delta )

    # Update weights with learning rate
    w1 += lr * l1.T.dot(l2_delta)
    w0 += lr * l0.T.dot(l1_delta)


print( "l2:", l2.shape, l2)
print( "bye" )