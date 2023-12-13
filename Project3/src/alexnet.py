import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from datetime import datetime

from data import trainloader, testloader, classes

EPOCHS = 20

# Based on the AlexNet architecture
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(-1, 256 * 6 * 6)
        x = self.classifier(x)
        return x

cnn = AlexNet()
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
