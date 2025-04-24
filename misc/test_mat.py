import numpy as np


def pp( obj ):
    print( f"{obj.ndim}d", "/", obj.shape, "  ", obj )


def calc_wx( W, x ):   ## use w,x for arguments
    ##  y = x*W      e.g    (1xn) (nxm) => (1xm)
    ##  y = W.T*x           (mxn) (nx1) => (mx1)
    ##
    ##   (x*W).T == W.T*x
    
    ## row vector  - one row, many columns (1xn)
    xrow = x.reshape( 1, -1 )
    print( "x (row vector - one row, many columns)", xrow.shape, xrow )
    #=>  (1,3)  [[1 2 3]]
    #=>  x (row vector - one row, many columns) (1, 3) [[1 2 3]]
    res1a1 = np.matmul( xrow, W ) 
    res1a2 = np.matmul( x, W )
    res1b1 = np.dot( xrow, W )
    res1b2 = np.dot( x, W )
    pp( res1a1 )
    pp( res1a2 )
    pp( res1b1 )
    pp( res1b2 )
    print( ".T" )
    pp( res1a1.T )
    pp( res1a2.T )
    pp( res1b1.T )
    pp( res1b2.T )

    ## column vector - many rows, one column (nx1)
    xcol = x.reshape( -1, 1 )
    print( "x (column vector - many rows, one column)", xcol.shape, xcol )
    #=>  x (column vector - many rows, one column) (3, 1) [[1]
    #                                                      [2]
    #                                                      [3]]

    res2a1 = np.matmul( W.T, xcol ) 
    res2a2 = np.matmul( W.T, x )
    res2b1 = np.dot( W.T, xcol )
    res2b2 = np.dot( W.T, x )
    pp( res2a1 )
    pp( res2a2 )
    pp( res2b1 )
    pp( res2b2 )
 
    assert np.array_equal( res1a1.T, res2a1 )
    assert np.array_equal( res1a2.T, res2a2 )
    assert np.array_equal( res1a2,   res2a2 )
    assert np.array_equal( res1b1.T, res2b1 )
    assert np.array_equal( res1b2.T, res2b2 )
    assert np.array_equal( res1b2, res2b2 )



def calc_bwx( W, X ):   ## try batch version!!!
    ## assume 
    ##    X is matrix (batch x inputs)

    x = X
    res1a1 = np.matmul( x, W ) 
    res1b1 = np.dot( x, W )
    pp( res1a1 )
    pp( res1b1 )
    print( ".T" )
    pp( res1a1.T )
    pp( res1b1.T )

    x = X.T
    res2a1 = np.matmul( W.T, x ) 
    res2b1 = np.dot( W.T, x )
    pp( res2a1 )
    pp( res2b1 )
 
    assert np.array_equal( res1a1.T, res2a1 )
    assert np.array_equal( res1b1.T, res2b1 )





x = np.array([1,2,3])    ## inputs (vec3)
W = np.array( [[1.1, 1.2, 1.3, 1.4],
               [2.1, 2.2, 2.3, 2.4],
               [3.1, 3.2, 3.3, 3.4]]
            )     ## weights (3x4) - 3 inputs, 4 outputs


calc_wx( W, x )
print( "---" )


X = np.array([[1,2,3],
              [4,5,6]] )
print( "-- batch:")
calc_bwx( W, X )

## try another
X = np.array([[0.1,0.2,0.3],
              [0.4,0.5,0.6],
              [0.7,0.8,0.9]] )
print( "-- batch:")
calc_bwx( W, X )


print( "bye")