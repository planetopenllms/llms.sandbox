import numpy as np
from mintorch import Tensor


np.random.seed(42)

## note - always use (mini-batch) for data (input/x) and 
##                                     target (label/y)
data   = Tensor( [[0,0],[0,1],[1,0],[1,1]] )
target = Tensor( [[0],[1],[0],[1]] )


## use consts - why? why not?
##  INPUT_DIM = 2
##  HIDDEN_DIM = 3
##  OUTPUT_DIM = 1

w = [ Tensor( np.random.rand(2,3), requires_grad=True ),
      Tensor( np.random.rand(3,1), requires_grad=True )
    ]


for epoch in range(20):
    # Predict mini-batch
    pred = data.mm(w[0]).mm(w[1])
    print( pred )
    
    # Compare
    ##   with pow(er) e.g. uses x*x instead of x**2
    ## loss = ((pred - target)*(pred - target)).sum(axis=0)
    loss = ((pred - target)**2).sum(axis=0)
    print( loss )

    # Learn
    loss.backward()

    ## note - grad are NOT wrapped tensors but are plain ndarrays!!!
    for w_ in w:
        w_.value -= w_.grad * 0.1
        w_.grad *= 0

    print( f"epoch {epoch+1} - loss: {loss}" )


print( "bye")