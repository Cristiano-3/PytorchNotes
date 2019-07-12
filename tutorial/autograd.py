# coding: utf-8

import torch

# Tensor and Function
#---------------------------------------------
x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2
print(y)
print(y.grad_fn)

z = y * y * 3
out = z.mean() 

print(z, out)

a = torch.randn(2, 2)
a = ((a * 3)/(a - 1))
print(a.requires_grad)  # False
a.requires_grad_(True)  
print(a.requires_grad)  # True
b = (a * a).sum()
print(b.grad_fn)  # if a requires_grad is False: None, else: SumBackward0


# Gradients
#---------------------------------------------
# out contains a single scalar, out.backward() is equivalent to 
# out.backward(torch.tensor(1.))
out.backward(torch.tensor(1.))  # out.backward()
print(x.grad)


# feed external gradients into a model that has non-scalar output.
x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)    


# v is dl/dy, compute dl/dx = dl/dy * dy/dx
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
print(x.grad)


# stop autograd from tracking history on Tensors with .requires_grad=True,
# by wrapping the code block in with torch.no_grad():
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)

