from mintorch import Tensor


a = Tensor(3, label='a')
b = Tensor(4, label='b')
c = a + b;  c.label = 'c'
d = c * c;  d.label = 'd'
e = d + a;  e.label = 'e'
f = e + 3;  f.label = 'f'

print( a )
print( b )
print( f )

f.backward()
print( a.grad, b.grad, c.grad, d.grad, e.grad, f.grad )
#=> 15 14 14 1 1 1


print( "bye" )