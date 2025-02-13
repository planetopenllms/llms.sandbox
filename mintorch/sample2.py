from mintorch import Tensor


a = Tensor([1,2,3,4,5])
b = Tensor([2,2,2,2,2])
c = Tensor([5,4,3,2,1])
print( a )
print( b )
print( c )

d = a + b
e = b + c
f = d + e
f.backward()

print( a.grad, b.grad, c.grad, d.grad, e.grad, f.grad )
print( f )
## b.grad.data == np.array([2,2,2,2,2])




print( "bye")