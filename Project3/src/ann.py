from collections import Counter
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim

from data import trainloader, testloader, classes

EPOCHS = 250
PATIENCE = 10  # Number of epochs with no improvement to wait before stopping
LEARNING_RATES = [0.001]
MOMENTA = [0.9]
DROPOUT_PROBS = [0.1, 0.3, 0.5, 0.7]

class DeepANN(nn.Module):
    def __init__(self):
        super(DeepANN, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 512)  # Increase units in the first hidden layer
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)  # Add a second hidden layer with 256 units
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 128)  # Add a third hidden layer with 128 units
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(128, 64)  # Add a fourth hidden layer with 64 units
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(64, 32)  # Add a fifth hidden layer with 32 units
        self.relu5 = nn.ReLU()
        self.fc6 = nn.Linear(32, 16)  # Add a sixth hidden layer with 16 units
        self.relu6 = nn.ReLU()
        self.fc7 = nn.Linear(16, 8)  # Add a seventh hidden layer with 8 units
        self.relu7 = nn.ReLU()
        self.fc8 = nn.Linear(8, 10)  # Output layer with 10 units

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)  # Flatten the input
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.relu4(self.fc4(x))
        x = self.relu5(self.fc5(x))
        x = self.relu6(self.fc6(x))
        x = self.relu7(self.fc7(x))
        x = self.fc8(x)
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

class ModifiedSimpleANN(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super(ModifiedSimpleANN, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 512)
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(512, 256)
        self.elu2 = nn.ELU()
        self.dropout2 = nn.Dropout(p=dropout_prob)
        self.fc3 = nn.Linear(256, 128)
        self.elu3 = nn.ELU()
        self.dropout3 = nn.Dropout(p=dropout_prob)
        self.fc4 = nn.Linear(128, 64)
        self.elu4 = nn.ELU()
        self.dropout4 = nn.Dropout(p=dropout_prob)
        self.fc5 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = self.dropout1(self.elu1(self.fc1(x)))
        x = self.dropout2(self.elu2(self.fc2(x)))
        x = self.dropout3(self.elu3(self.fc3(x)))
        x = self.dropout4(self.elu4(self.fc4(x)))
        x = self.fc5(x)
        return x

def main():
    best = 0
    best_dropout, best_lr, best_momentum = [], [], []
    for lr in LEARNING_RATES:
        for momentum in MOMENTA:
            for dropout_prob in DROPOUT_PROBS:
                print("Running model with lr={}, momentum={}, dropout_prob={}".format(lr, momentum, dropout_prob))
                accuracy = run_model(lr, momentum, dropout_prob)
                if accuracy > best:
                    best = accuracy
                    best_dropout.append(dropout_prob)
                    best_lr.append(lr)
                    best_momentum.append(momentum)

    # Print the most recent 'best' values
    print("Most Recent Best Values:")
    print(f"Best Dropout: {best_dropout[-1]}")
    print(f"Best Learning Rate: {best_lr[-1]}")
    print(f"Best Momentum: {best_momentum[-1]}")
    print("")

    # Print the most common 'best' values
    print("Most Common Best Values:")
    print(f"Most Common Dropout: {Counter(best_dropout).most_common(1)[0][0]}")
    print(f"Most Common Learning Rate: {Counter(best_lr).most_common(1)[0][0]}")
    print(f"Most Common Momentum: {Counter(best_momentum).most_common(1)[0][0]}")
    return

def run_model(lr, momentum, dropout_prob):
    ann = ModifiedSimpleANN(dropout_prob)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ann.parameters(), lr=lr, momentum=momentum)

    start_time = time.time()

    # Save the best model parameters
    best_model_params = copy.deepcopy(ann.state_dict())
    best_validation_loss = float('inf')
    no_improvement_count = 0

    print("Training for {} epochs".format(EPOCHS))
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