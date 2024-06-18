# 3rd parties
import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

# 1st parties
import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_mnist_data(batch_size=32, target_dir='./data'):
    """
    Downloading and loading the MNIST dataset

    :param batch_size: The batch size for the data loader
    :param target_dir: The directory where the data will be downloaded
    :return: The train and test data loaders
    """
    train_set = torchvision.datasets.MNIST(
        root=target_dir, train=True, download=True, transform=torchvision.transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_set = torchvision.datasets.MNIST(
        root=target_dir, train=False, download=True, transform=torchvision.transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

def test_model(model, testloader, criterion):
    """
    Testing (evaluation mode - No backpropagation) the model on the given testing set.
    :param model: The model to test.
    :param testloader: The testing set.
    :param criterion: The loss function to use.
    :return: The loss and accuracy of the model on the testing set.
    """
    print('[TEST] Starting evaluation...')
    model.eval()
    with torch.no_grad():
        ep_loss = 0
        
        for i, (inputs, labels) in enumerate(tqdm(testloader)):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, inputs)

            ep_loss += loss.item()

        loss = ep_loss / len(testloader)

        return loss

def train_model(
        model, 
        train_loader, 
        test_loader, 
        optimizer, 
        loss_criterion, 
        epochs):
    """
    Training the model on the training set.
    :param model: The model to train.
    :param trainloader: The training set.
    :param test_loader: The testing set.
    :param epochs: The number of epochs to train the model.
    :param lr: The learning rate for the optimizer.
    :return: The losses and accuracies of the model during training, for each epoch.
    """

    train_losses = []
    test_losses = []

    for ep in range(epochs):

        print(f'[TRAIN] Epoch: {ep + 1}')
        model.train()
        ep_loss = 0.

        for i, (inputs, labels) in enumerate(tqdm(train_loader)):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_criterion(outputs, inputs)
            loss.backward()
            optimizer.step()

            ep_loss += loss.item()

        train_losses.append(ep_loss / len(train_loader))

        # Making an evaluation run on the current state        
        test_losses.append(test_model(model, test_loader, loss_criterion))

    print('[TRAIN] Final Loss: ', train_losses[-1])

    return train_losses, test_losses

def main():
    # Loading the MNIST dataset (first run may take a little while more)
    train, test = load_mnist_data()

    ### Q1 - Training & evaluating the simple MNIST AE model

    # Hyperparameters
    lr = 0.01
    epochs = 5

    ae_model = models.ConvAutoencoder()
    ae_model.to(device)
    print(f'[DEBUG] Using device: {device}')
    optimizer = torch.optim.Adam(ae_model.parameters(), lr=lr)
    loss_criterion = torch.nn.L1Loss().to(device)

    try:
        ae_model.load_state_dict(torch.load('ae_model.pth'))
    except FileNotFoundError:
        train_losses, test_losses = train_model(ae_model, train, test, optimizer, loss_criterion, epochs)
        torch.save(ae_model.state_dict(), 'ae_model.pth')
        
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.legend()
    plt.show()

    ### Q2

if __name__ == "__main__":
    main()