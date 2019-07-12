# coding: utf-8

import torch

'''
# move model and tensors to GPU
device = torch.device('cuda:0')
model.to(device)

# run on multi-GPUS
model = nn.DataParallel(model)
'''

# import modules and parameters
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

input_size = 5
output_size = 2

batch_size = 30
data_size = 100

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# virtual dataset
class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size), 
                        batch_size=batch_size, shuffle=True) 
                        

# simple model
class Model(nn.Module):
    # our model
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print('\tIn Model: input size', input.size(), 'output size', output.size())
        return output


# create a model and use dataparallel, put it on GPUs 
model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)  # use _count GPUs

model.to(device)  # use _count GPUs


# run model
for data in rand_loader:
    input = data.to(device)
    output = model(input)
    print("Outside: input_size", input.size(), "output_size", output.size())