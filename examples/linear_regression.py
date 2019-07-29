# coding: utf-8

import torch
import numpy as np

# Simple Liner Regression 
# Fit a line to the data. Y =w.x+b 
np.random.seed(0)
torch.manual_seed(0)

# Step 1: Dataset
w = 2
b = 3

x = np.linspace(0, 10, 100)
y = w*x + b + np.random.randn(100)*2

xx = x.reshape(-1, 1)
yy = y.reshape(-1, 1)

# Step 2: Model
class LinearRegressionModel(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(LinearRegressionModel, self).__init__()
        self.model = torch.nn.Linear(D_in, D_out)

    def forward(self, x):
        y_pred = self.model(x)    
        return y_pred


model = LinearRegressionModel(D_in=1, D_out=1)

# Step 3: Training
cost = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
inputs = torch.from_numpy(xx.astype("float32"))
outputs = torch.from_numpy(yy.astype("float32"))

for epoch in range(100):
    # 3.1 forward pass
    y_pred = model(inputs)

    # 3.2 compute loss
    loss = cost(y_pred, outputs)

    # 3.3 backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1)%10 == 0:
        print("epoch{}, loss{}".format(epoch+1, loss.data))


# Step 4: Display model and confirm 
import matplotlib.pyplot as plt 
plt.figure(figsize=(4,4)) 
plt.title("Model and Dataset") 
plt.xlabel("X")
plt.ylabel("Y") 
plt.grid() 
plt.plot(x,y,"ro",label="DataSet",marker="x",markersize=4) 
plt.plot(x,model.model.weight.item()*x+model.model.bias.item(),label="Regression Model") 
plt.legend();plt.show()

# reference: https://flashgene.com/archives/51458.html