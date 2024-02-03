import os

import torch
import torch.optim as optim

import matplotlib.pyplot as plt

from utils import Net, train, test, full_train_dataset, test_dataset
from params import RANDOM_SEED, LEARNING_RATE, MOMENTUM, models_path, BATCH_SIZE_TEST, BATCH_SIZE_TRAIN

torch.backends.cudnn.enabled = False
torch.manual_seed(0)

total_number_of_split_data = 3

model_number = 3
model_name = f'model_v{model_number}'
n_epochs = 5

if __name__ == '__main__':
    # Get model path
    model_path = os.path.join(models_path, model_name)
    os.makedirs(model_path, exist_ok=True)

    # Split data for local models TODO: dodać opcje sredniej wazonej
    data_sizes = [int(1 / total_number_of_split_data * len(full_train_dataset)) for _ in
                  range(total_number_of_split_data)]
    splitted_datasets = torch.utils.data.random_split(full_train_dataset, data_sizes)

    # Prepare train data for specific local model
    try:
        train_dataset = splitted_datasets[model_number - 1]
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=BATCH_SIZE_TRAIN,
                                                   shuffle=True)
    except IndexError:
        raise IndexError(f'There are {total_number_of_split_data} total numbers of split data!'
                         f'You put {model_number}!')

    # Prepare test data, independent to local model
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=BATCH_SIZE_TEST,
                                              shuffle=True)

    # Define network and optimizer
    network = Net()
    network_optimizer = optim.SGD(network.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    # Train and test model
    train_losses, train_counter = [], []
    test_losses, test_counter = [], [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

    test(model=network, test_loader=test_loader, test_losses=test_losses)
    for epoch in range(1, n_epochs + 1):
        train(model=network,
              optimizer=network_optimizer,
              epoch=epoch,
              train_loader=train_loader,
              train_losses=train_losses,
              train_counter=train_counter,
              model_path=model_path,
              model_name=model_name, )
        test(model=network, test_loader=test_loader, test_losses=test_losses)

    # Plot loss function
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()
