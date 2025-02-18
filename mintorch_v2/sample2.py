import numpy as np
from mintorch import Tensor
from nn import SGD, Sequential, Linear


np.random.seed(42)

## note - always use (mini-batch) for data (input/x) and 
##                                     target (label/y)
x  = Tensor( [[0,0],[0,1],[1,0],[1,1]] )
y  = Tensor( [[0],[1],[0],[1]] )


## use consts - why? why not?
##  INPUT_DIM = 2
##  HIDDEN_DIM = 3
##  OUTPUT_DIM = 1

# w = [ Tensor( np.random.rand(2,3)),
#      Tensor( np.random.rand(3,1))
#    ]
#   optim = SGD( parameters=w, lr=0.1 )


model = Sequential( [Linear(2,3), 
                     Linear(3,1)])

optim = SGD( parameters=model.get_parameters(), lr=0.05 )



for epoch in range(20):
    # Predict mini-batch
    y_pred = model.forward( x )
    print( y_pred )
    
    # Compare
    ##   with pow(er) e.g. uses x*x instead of x**2
    ## loss = ((pred - target)*(pred - target)).sum(axis=0)
    loss = ((y_pred - y)**2).sum(axis=0)
    print( loss )

    # Learn
    loss.backward()
    optim.step()
    print( f"epoch {epoch+1} - loss: {loss}" )


print( "bye")