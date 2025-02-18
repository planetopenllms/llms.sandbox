import numpy as np
from mintorch import Tensor
from nn import SGD, Sequential, Linear, Tanh, Sigmoid, MSELoss


np.random.seed(42)

## note - always use (mini-batch) for data (input/x) and 
##                                     target (label/y)
x  = Tensor( [[0,0],[0,1],[1,0],[1,1]] )
y  = Tensor( [[0],[1],[0],[1]] )



model = Sequential( [Linear(2,3), Tanh(),
                     Linear(3,1), Sigmoid()])

criterion = MSELoss()
optim = SGD( parameters=model.get_parameters(), lr=1 )



for epoch in range(20):
    # Predict mini-batch
    y_pred = model.forward( x )
    print( y_pred )
    
    # Compare
    loss = criterion.forward( y_pred,  y)
    print( loss )

    # Learn  (calc gradients)
    loss.backward()
    optim.step()      ## update parameter (weights)
    print( f"epoch {epoch+1} - loss: {loss}" )


print( "bye")