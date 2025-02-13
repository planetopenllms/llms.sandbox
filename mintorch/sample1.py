from mintorch import Tensor


x = Tensor([1,2,3,4,5])
y = Tensor([2,2,2,2,2])
print( x )
print( y )
 
z = x + y
print( z )
z.backward()
 
print( x.grad, y.grad, z.grad )



a = Tensor([1,2,3,4,5])
b = Tensor([2,2,2,2,2])
c = Tensor([5,4,3,2,1])
d = Tensor([-1,-2,-3,-4,-5])
 
e = a + b
f = c + d
g = e + f
 
g.backward()

print(a.grad, b.grad, c.grad, d.grad, e.grad, f.grad, g.grad )



print( "bye")