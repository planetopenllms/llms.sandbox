from mintorch import Tensor


##
##  f(x,y)  =  x**2 * y + y + 2
##             (x*x) * y + y + 2
##
##  x=3, y=4
##  f(3,4) = 42
##  df_dx = 24, df_dy = 10
def test_homl1():
    x = Tensor(3)
    y = Tensor(4)

    ## f = (x*x) * y + y + Tensor(2)
    f = x**2 * y + y + 2

    print( x )
    print( y )
    print( f )

    f.backward()

    print( x.grad, y.grad )
    print( f )

    assert f.data == 42 
    assert x.grad == 24 
    assert y.grad == 10    


##
##  f(w1,w2)  =  3w1**2 + 2w1*w2
##
##  w1=5, w2=3
##  f(5,3) = 105
##  df_dw1 = 36, df_dw2 = 10
def test_homl2():
    w1 = Tensor(5)
    w2 = Tensor(3)

    f = 3*w1**2 + 2*w1*w2

    print( w1 )
    print( w2 )
    print( f )

    f.backward()

    print( w1.grad, w2.grad )
    print( f )

    assert( f.data == 105 )
    assert( w1.grad == 36 )
    assert( w2.grad == 10 )
