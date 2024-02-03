import os
import re

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from params import LOG_INTERVAL


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


full_train_dataset = torchvision.datasets.MNIST('data/files/', train=True, download=True,
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                                ]))

test_dataset = torchvision.datasets.MNIST('data/files/', train=False, download=True,
                                          transform=torchvision.transforms.Compose([
                                              torchvision.transforms.ToTensor(),
                                              torchvision.transforms.Normalize(
                                                  (0.1307,), (0.3081,))
                                          ]))


def train(model, optimizer, epoch: int, train_loader, train_losses: list, train_counter: list, model_path: str,
          model_name: str) -> None:
    """
    Trains the model for a single epoch and saves the model and optimizer states.
    The function iterates over the training data, performs a forward pass, computes the loss,
    performs a backward pass, updates the model weights, and saves the model and optimizer states
    after training. It also prints training progress and loss.

    :param model: The neural network model to be trained.
    :param optimizer: The optimization algorithm used to update the model weights.
    :param epoch: The current training epoch.
    :param train_loader: DataLoader for the training dataset.
    :param train_losses: List to record the loss of each training step.
    :param train_counter: List to record the number of examples seen during training.
    :param model_path: Path where the model and optimizer states are saved.
    :param model_name: Base name for saving the model and optimizer state files.
    """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({round(100. * batch_idx / len(train_loader), 1)}%)]\tLoss: {round(loss.item(), 3)}')
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))

    torch.save(model.state_dict(), os.path.join(model_path, f'{model_name}_epoch_{epoch}.m.pth'))


def test(model, test_loader, test_losses: list) -> None:
    """
    Evaluates the model on the test dataset.
    The function iterates over the test data, performs a forward pass, computes the loss,
    and calculates the total number of correct predictions. It prints the average loss and
    the accuracy of the model on the test dataset.

    :param model: The neural network model to be evaluated.
    :param test_loader: DataLoader for the test dataset.
    :param test_losses (list): List to record the loss of each testing step.
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print(f'\nTest set: Avg. loss: {round(test_loss, 3)}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({round(float(100. * correct / len(test_loader.dataset)), 1)}%)\n')


def extract_epoch_number(filename):
    """
    Extracts the epoch number from a model file name.

    :param filename: The name of the model file from which the epoch number is to be extracted.
    :return: The extracted epoch number from the file name, or 0 if the epoch number cannot be found.
    """
    match = re.search(r'epoch_(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0


#
def print_pretty(models_dict) -> None:
    """
    Formatting and displaying the dictionary in a more readable way.

    :param models_dict: model dict to print
    """
    for model_name, files in models_dict.items():
        print(f"{model_name}:")
        files.sort()
        for file in files:
            print(f"  - {file}")
    print('\n')
