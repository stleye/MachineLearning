
import numpy as np
import torch
import helper
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

def activation(x):
	return 1/(1+torch.exp(-x))

def softmax(x):
	return torch.exp(x) / (torch.sum(torch.exp(x), dim=1).view(x.shape[0], -1))

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = next(dataiter)

inputs = images.view(images.shape[0], -1)

n_input = inputs.shape[1]
n_hidden = 256
n_output = 10

W1 = torch.randn(n_input, n_hidden)
B1 = torch.randn(1, n_hidden)

W2 = torch.randn(n_hidden, n_output)
B2 = torch.randn(1, n_output)

h = activation(torch.mm(inputs, W1) + B1)

out = torch.mm(h, W2) + B2

probabilities = softmax(out)

print(probabilities.shape)
print(probabilities.sum(dim=1))
