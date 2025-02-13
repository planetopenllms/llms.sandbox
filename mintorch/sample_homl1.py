from mintorch import Tensor

##
##  f(x,y)  =  x**2 * y + y + 2
##             (x*x) * y + y + 2
##
##  x=3, y=4
##  f(3,4) = 42
##  df_dx = 24, df_dy = 10



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



print( "bye")