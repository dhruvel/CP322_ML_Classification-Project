import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset

# Ensure we're running from src directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))

_transform = transforms.Compose([transforms.ToTensor()])
_trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=_transform)

# Stack all train images together into a tensor
x = torch.stack([sample[0] for sample in ConcatDataset([_trainset])])
# Get the mean and std of each channel
mean = torch.mean(x, dim=(0, 2, 3))
std = torch.std(x, dim=(0, 2, 3))

# Load data again, this time with normalizing
_transform = transforms.Compose([
    # transforms.Resize(227),     # Comment out for ANN
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
    # Could add rotation and/or cropping here to add more samples to train on
])

# Increasing batch size reduces training time, but might also reduce accuracy
batch_size = 64

# Load train data
_trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=_transform)
trainloader = torch.utils.data.DataLoader(_trainset, batch_size=batch_size, shuffle=True, num_workers=0)

# Load test data
_testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=_transform)
testloader = torch.utils.data.DataLoader(_testset, batch_size=batch_size, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print('Training data size:', len(trainloader.dataset))
print('Test data size:', len(testloader.dataset))
