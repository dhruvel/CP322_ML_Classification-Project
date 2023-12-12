import os
import torch
import torchvision
import torchvision.transforms as transforms

# Ensure we're running from src directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Normalize the data
_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# PyTorch recommends a batch size of 4, but I feel that's too small
# Increasing batch size reduces training time, but might also reduce accuracy
batch_size = 32

# Load train data
_trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=_transform)
trainloader = torch.utils.data.DataLoader(_trainset, batch_size=batch_size, shuffle=True, num_workers=4)

# Load test data
_testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=_transform)
testloader = torch.utils.data.DataLoader(_testset, batch_size=batch_size, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print('Training data size:', len(trainloader.dataset))
print('Test data size:', len(testloader.dataset))
