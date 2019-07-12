# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F 


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # kernels
        self.conv1 = nn.Conv2d(1, 6, 3)  # in_channel, out_channel, kernal_size, stride=1
        self.conv2 = nn.Conv2d(6, 16, 3)

        # an affine op: y=Wx+b
        self.fc1 = nn.Linear(16*6*6, 120)  # 6*6 infered from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        # max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # dimensions except channel    
        num_features = 1
        for s in size:
            num_features *= s

        return num_features    


# print network 
net = Net()
print(net)

params = list(net.parameters())
print(len(params), type(params[0]), params[0])
print(params[0].size())  # conv1's weight

# try random 32x32 input
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

# zero the gradient buffers of all params
# and backprops with random gradient
net.zero_grad()
out.backward(torch.randn(1, 10))

# Loss
output = net(input)
target = torch.randn(10)     # a dummy target
target = target.view(1, -1)  # same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

# BackProp
# To backpropagate the error all we have to do is to loss.backward(). 
# You need to clear the existing gradients though, else gradients will be accumulated to existing gradients.
net.zero_grad()  # zeroes the gradient buffers of all parameters
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()
print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

# Update the weights
# method 1
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

# method 2
import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=0.01) # create optimizer
optimizer.zero_grad()  # zero the gradient buffers

output = net(input)    
loss = criterion(output, target)

loss.backward()   # compute grads
optimizer.step()  # does the update
