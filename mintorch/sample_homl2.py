from mintorch import Tensor

##
##  f(w1,w2)  =  3w1**2 + 2w1*w2
##
##  w1=5, w2=3
##  f(5,3) = 105
##  df_dw1 = 36, df_dw2 = 10



w1 = Tensor(5)
w2 = Tensor(3)

f = 3*w1**2 + 2*w1*w2


print( w1 )
print( w2 )
print( f )

f.backward()

print( w1.grad, w2.grad )
print( f )



print( "bye")