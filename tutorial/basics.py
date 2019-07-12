# coding: utf-8

import torch

# Tensors (type: torch.Tensor)
# ----------------------------------------------------------------
x = torch.empty(5, 3)  # Construct a matrix uninitialized matrix
print(x)

x = torch.rand(5, 3)   # Construct a matrix randomly initialized
print(x)

x = torch.zeros(5, 3, dtype=torch.long)  # matrix filled with zeros of dtype long
print(x)

x = torch.Tensor([5.5, 3])  # Construct a tensor directly from data
print(x)

x = x.new_ones(5, 3, dtype=torch.double)  # same type... as x, otherwise overwrite dtype... 
print(x)

x = torch.randn_like(x, dtype=torch.float)  # overwrite x's dtype
print(x, x.size())  # torch.Size([5, 3])

# Operations
# ----------------------------------------------------------------
## addition
y = torch.rand(5, 3)
print(x + y)
print(torch.add(x, y))
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)
y.add_(x)  # adds x to y, in-place
print(y)
print(x[:, 1])  # use standard NumPy-like indexing with all bells and whistles!

x = torch.randn(4, 4)
y = x.view(16)  # if you want to resize/reshape tensor, you can use torch.view
z = x.view(-1, 8)  # -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

# If you have a one element tensor, use .item() to get the value as a Python number
x = torch.randn(1)
print(x)
print(x.item())


# Numpy Bridge
# The Torch Tensor and NumPy array will share their underlying memory locations
# (if the Torch Tensor is on CPU), and changing one will change the other.
# -----------------------------------------------------------------
## tensors2ndarrays
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)

## ndarrays2tensors
import numpy as np 
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

# Cuda Tensors
# # ----------------------------------------------------------------
# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!