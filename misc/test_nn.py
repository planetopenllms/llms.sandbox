import numpy as np


x = np.array([1,2,3])    ## inputs (vec3)
print( x.shape, x )
w = np.array( [[1.1, 1.2, 1.3, 1.4],
               [2.1, 2.2, 2.3, 2.4],
               [3.1, 3.2, 3.3, 3.4]]
            )     ## weights (3x4) - 3 inputs, 4 outputs
print( w.shape, w )
print( w.T.shape, w.T )
print( w.T.T.shape, w.T.T )


## row vector  - one row, many columns (1,n)
xrow = x.reshape( 1, -1 )
print( xrow.shape, xrow )
#=>  (1,3)  [[1 2 3]]

## column vector - many rows, one column (n,1)
xcol = x.reshape( -1, 1 )
print( xcol.shape, xcol )
#=>  (3, 1) [[1]
#            [2]
#            [3]]

## try transform
print( xrow.T.shape, xrow.T )
print( xcol.T.shape, xcol.T )


print( "---" )




def test_matmul():
   assert np.array_equal( w, w.T.T )

   z1 = np.dot( x, w )
   print( z1.shape, z1 )
   z2 = np.dot( w.T, x )
   print( z2.shape, z2 )

   assert np.array_equal( z1, z2 )

   assert np.array_equal( xcol, xrow.T )
   assert np.array_equal( xrow, xcol.T )
   assert np.array_equal( xrow, xrow.T.T )
   assert np.array_equal( xcol, xcol.T.T )


   ##  note - matmul WITH vector possible ( will NOT enforce matrix) !!!
   #               vect_matmul or mat_vectmul  
   z1a = np.matmul( x, w )
   print( "z1a:", z1a.shape, z1a )
   assert np.array_equal( z1, z1a )

   z2a = np.matmul( w.T, x )
   print( "z2a:", z2a.shape, z2a )
   assert np.array_equal( z2, z2a )

   z1b = np.matmul( xrow, w )       # shape is (1,3)@(3,4) => (1,4)
   print( "z1b:", z1b.shape, z1b )
   z2b = np.matmul( w.T, xcol )     # shape is (4,3)@(3,1) => (4,1)  
   print( "z2b:",  z2b.shape, z2b )              
   print( "z2b.T:", z2b.T.shape, z2b.T )   #  (4,1).T => (1,4) AND
                                           #  (1,4).T => (4,1)

   assert np.array_equal( z1b, z2b.T )
   assert np.array_equal( z2b, z1b.T )

   ## note - sum/average etc. operate on ALL elements resulting in scalar!!!
   ##       use np.sum( .., axis=0|1|...)  to sum only rows, cols, etc.
   print( np.sum( z1 ), np.sum( z2) )   ## sums ALL elements
   print( np.average( z1), np.average( z2 ))  ## averages ALL elements


def test_e():
    ## test e notation for numbers
    print( 0.1, 1e-1, 1e-01, 10**-1, 1/10, 1/10**1 )

    assert 0.1 == 1e-1
    assert 0.1 == 1e-01
    assert 0.1 == 10**-1
    assert 0.1 == 1/10
    assert 0.1 == 1/10**1

    print( 0.01,  1e-2, 1e-02, 10**-2, 1/100, 1/10**2 )
    print( 0.001, 1e-3, 1e-03, 10**-3, 1/1000, 1/10**3 )

    assert 0.001 == 1e-3
    assert 0.001 == 1e-03
    assert 0.001 == 10**-3
    assert 0.001 == 1/1000
    assert 0.001 == 1/10**3

    print( 0.42, 42e-2, 42e-02, 42*10**-2, 42/100, 42/10**2 )
    assert 0.42 == 42e-2
    assert 0.42 == 42e-02
    assert 0.42 == 42*10**-2
    assert 0.42 == 42/100
    assert 0.42 == 42/10**2


def test_math():
####
#  some more equal operations
#      square root 
#        is    x**0.5
#      division
##       is    x/y =  x*y**-1  

    print( "np.sqrt( 25 )", np.sqrt( 25 ), 25**0.5 )    #=>   5  / 5
    assert np.sqrt( 25 ) == 25**0.5
    assert np.sqrt( 25 ) == np.pow( 25, 0.5 )

    print( "1/np.sqrt( 25 )", 1/np.sqrt( 25 ), np.pow( 25, -0.5 ) )
    assert 1/np.sqrt( 25 ) == np.pow( 25, -0.5 )   ## e.g.  1/sqrt(25) = 25**-0.5
    assert 1/np.sqrt( 25 ) == 25**-0.5 
    

    print( 25 / 5, 25 * 5**-1, 5**-1 )    #=> 5 / 5 / 0.2  
    assert 25 / 5 == 25 * 5**-1


def test_error():
   ##  y - y_hat  ==   -(y_hat - y)
   ##  note  - if you calculate the error
   ##           you can use both formulas/ways
   ##           the result only differs in the sign (e.g. 2 vs -2, 1.79 vs -1.79 etc.)
   samples = [(20, 18),            #=>   2 / -2
              (19.9, 18.1),        #=>   1.7999999999999972 / -1.7999999999999972
              (0.8, -0.4)]         #=>   1.2000000000000002 / -1.2000000000000002
   for y, y_hat in samples: 
       print( y-y_hat, y_hat-y )
       assert y-y_hat == -(y_hat-y)



if __name__ == "__main__":
    test_matmul()
    test_e()
    test_math()
    test_error()
    print( "bye")