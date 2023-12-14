from collections import Counter
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import prune

from data import trainloader, testloader, classes

EPOCHS = 250
PATIENCE = 80
HIDDEN_LAYERS = [
    [512, 256, 128]
]

class DeepANN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(DeepANN, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size

        layers = []
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(1, len(sizes)):
            linear_layer = nn.Linear(sizes[i - 1], sizes[i])
            layers.append(linear_layer)
            layers.append(nn.BatchNorm1d(sizes[i]))
            if i < len(sizes) - 1:
                layers.append(nn.ReLU())

            # Add pruning to the linear layer
            prune.l1_unstructured(linear_layer, name='weight', amount=0.2)

        # Create the model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Flatten the input if needed
        input_size = x.view(x.size(0), -1).size(1)
        x = x.view(-1, input_size)
        x = self.model(x)
        return x

class SimpleANN(nn.Module):
    def __init__(self):
        super(SimpleANN, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 256)
        self.leaky1 = nn.LeakyReLU()  # Use elu activation
        self.fc2 = nn.Linear(256, 128)
        self.leaky2 = nn.LeakyReLU()  # Use elu activation
        self.fc3 = nn.Linear(128, 64)
        self.leaky3 = nn.LeakyReLU()  # Use elu activation
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = self.leaky1(self.fc1(x))
        x = self.leaky2(self.fc2(x))
        x = self.leaky3(self.fc3(x))
        x = self.fc4(x)
        return x

def main():
    best = 0
    func = None
    for hidden_layer in HIDDEN_LAYERS:
        accuracy = run_model(hidden_layer)
        best = accuracy if accuracy > best else best
        func = hidden_layer if func == None or accuracy > best else func

    print('best is: ', best)
    print("layer is", func)
    return

def run_model(hidden_sizes):
    print("hidden layer is:", hidden_sizes)
    input_size = 32 * 32 * 3
    output_size = 10
    ann = DeepANN(input_size, hidden_sizes, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ann.parameters(), lr=0.001, momentum=0.9, weight_decay = 1e-5)

    start_time = time.time()

    # Save the best model parameters
    best_model_params = copy.deepcopy(ann.state_dict())
    best_validation_loss = float('inf')
    no_improvement_count = 0

    for epoch in range(EPOCHS):
        print("Epoch {}".format(epoch + 1))

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            # Forward + Backpropagate + Gradient Descent
            outputs = ann(inputs)
            loss = criterion(outputs, labels)
            torch.nn.utils.clip_grad_norm_(ann.parameters(), max_norm=1.0)
            loss.backward()
            optimizer.step()

        # Validation Loss
        ann.eval()
        with torch.no_grad():
            validation_loss = 0.0
            for data in testloader:
                images, labels = data
                outputs = ann(images)
                loss = criterion(outputs, labels)
                validation_loss += loss.item()

        # Check for improvement in validation loss
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            no_improvement_count = 0
            # Save the new best model parameters
            best_model_params = copy.deepcopy(ann.state_dict())
        else:
            no_improvement_count += 1

        # Check for early stopping
        if no_improvement_count >= PATIENCE:
            print("Early stopping: No improvement for {} epochs".format(PATIENCE))
            break

        print('Validation loss: {:.3f}'.format(validation_loss))

    print('Finished Training in {} seconds'.format(time.time() - start_time))

    # Load the best model parameters
    ann.load_state_dict(best_model_params)
    torch.save(ann.state_dict(), "models/ann.pt")
    ann.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = ann(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: {}'.format(100 * correct / total))
    return (100 * correct / total)

if __name__ == "__main__":
    main()