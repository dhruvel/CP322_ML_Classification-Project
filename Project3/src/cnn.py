import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from datetime import datetime

from data import trainloader, testloader, classes

EPOCHS = 1

CL3_64 = 0
CL64_128 = 1
CL128_128 = 2
CL128_256 = 3
CL256_256 = 4
CL256_512 = 5
CL512_2048 = 6
CL2048_256 = 7

BN64 = 0
BN128 = 1
BN256 = 2
BN512 = 3
BN2048 = 4

convLayers = [
    (3, 64, 3),
    (64, 128, 3),
    (128, 128, 3),
    (128, 256, 3),
    (256, 256, 3),
    (256, 512, 3),
    (512, 2048, 1),
    (2048, 256, 1),
]

# Based on the SimpleNet architecture
# https://arxiv.org/pdf/1608.06037v8.pdf
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.convs = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=(1 if kernel_size == 3 else 0))
            for in_channels, out_channels, kernel_size in convLayers
        ]
        self.norms = [
            nn.BatchNorm2d(out_channels, eps=0.00001, momentum=0.05, affine=True)
            for out_channels in [64, 128, 256, 512, 2048]
        ]
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.1)
        self.fc = nn.Linear(256, 10)
    
    def forward(self, x):
        x = F.relu(self.norms[BN64](self.convs[CL3_64](x)))             # 32x32x3 -> 32x32x64
        x = F.relu(self.norms[BN128](self.convs[CL64_128](x)))          # 32x32x64 -> 32x32x128
        for _ in range(2):                                              # 32x32x128 -> 32x32x128 x2
            x = F.relu(self.norms[BN128](self.convs[CL128_128](x)))
        x = self.pool(x)                                  # 32x32x128 -> 16x16x128
        for _ in range(2):                                              # 16x16x128 -> 16x16x128 x2
            x = F.relu(self.norms[BN128](self.convs[CL128_128](x)))
        x = F.relu(self.norms[BN256](self.convs[CL128_256](x)))         # 16x16x128 -> 16x16x256
        x = self.pool(x)                                  # 16x16x256 -> 8x8x256
        for _ in range(2):                                              # 8x8x256 -> 8x8x256 x2
            x = F.relu(self.norms[BN256](self.convs[CL256_256](x)))
        x = self.pool(x)                                  # 8x8x256 -> 4x4x256
        x = F.relu(self.norms[BN512](self.convs[CL256_512](x)))         # 4x4x256 -> 4x4x512
        x = self.pool(x)                                  # 4x4x512 -> 2x2x512
        x = F.relu(self.norms[BN2048](self.convs[CL512_2048](x)))       # 2x2x512 -> 2x2x2048
        x = F.relu(self.norms[BN256](self.convs[CL2048_256](x)))        # 2x2x2048 -> 2x2x256
        x = F.relu(self.norms[BN256](self.convs[CL256_256](x)))         # 2x2x256 -> 2x2x256
        x = self.pool(x)                                  # 2x2x256 -> 1x1x256
        x = x.view(-1, 256)                                             # 1x1x256 -> 256
        x = self.fc(x)                                                  # 256 -> 10
        return x

cnn = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)

start_time = time.time()

print("Training for {} epochs".format(EPOCHS))
for epoch in range(EPOCHS):
    print("Epoch {}".format(epoch + 1))

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        # Forward + Backpropagate + Gradient Descent
        outputs = cnn(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        # Current training loss and accuracy
        running_loss += loss.item()
        if (i + 1) % 100 == 0:
            accuracy = (torch.max(outputs, 1)[1] == labels).sum().item() / labels.size(0)
            print('[%d, %4d] loss: %.3f, accuracy: %.2f' % (epoch + 1, i + 1, running_loss / 100, accuracy * 100))
            running_loss = 0.0

print('Finished Training in {} seconds'.format(time.time() - start_time))

torch.save(cnn.state_dict(), "../models/cnn_{}.pt".format(datetime.now().strftime("%d%m%Y_%H%M%S")))

# Set model to evaluation mode
cnn.eval()

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = cnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: {}%'.format(100 * correct / total))
