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
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(
                                                    mean=[0.5], std=[0.5])])
    # transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
    #     torchvision.transforms.Pad(padding=2, fill=0, padding_mode='constant')])
    train_set = torchvision.datasets.MNIST(
        root=target_dir, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_set = torchvision.datasets.MNIST(
        root=target_dir, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader

def ae_evaluate(model, testloader, criterion):
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

def ae_train(
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
        print(f'[TRAIN] Loss: {train_losses[-1]}')

        # Making an evaluation run on the current state        
        test_losses.append(ae_evaluate(model, test_loader, loss_criterion))
        print(f'[TEST] Loss: {test_losses[-1]}')

    return train_losses, test_losses

def encoder_classifier_evaluate(model, testloader, criterion):
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
        correct = 0

        for i, (inputs, labels) in enumerate(tqdm(testloader)):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)

            ep_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        loss = ep_loss / len(testloader)
        accuracy = correct / len(testloader.dataset)

        return loss, accuracy
    
def encoder_classifier_train(model, train_loader, test_loader, optimizer, loss_criterion, epochs):
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
    train_accuracy = []
    test_losses = []
    test_accuracy = []

    for ep in range(epochs):

        print(f'[TRAIN] Epoch: {ep + 1}')
        model.train()
        ep_loss = 0.
        ep_correct = 0

        for i, (inputs, labels) in enumerate(tqdm(train_loader)):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            ep_loss += loss.item()
            ep_correct += (outputs.argmax(1) == labels).sum().item()

        train_losses.append(ep_loss / len(train_loader))
        train_accuracy.append(ep_correct / len(train_loader.dataset))
        print(f'[TRAIN] Loss: {train_losses[-1]}, Accuracy: {train_accuracy[-1]}')

        # Making an evaluation run on the current state        
        test_loss, test_acc = encoder_classifier_evaluate(model, test_loader, loss_criterion)
        test_losses.append(test_loss)
        test_accuracy.append(test_acc)
        print(f'[TEST] Loss: {test_losses[-1]}, Accuracy: {test_accuracy[-1]}')

    return train_losses, train_accuracy, test_losses, test_accuracy


def q1_ae_reconstruction(trainset, testset):
    """
    Experimenting with a simple MNIST convolutional autoencoder.
    The experiment visualizes the training/test losses and the reconstructed images.
    
    :param trainset: The training set.
    :param testset: The testing set.
    """
    # Hyperparameters
    lr = 0.01
    epochs = 10

    ae_model = models.ConvAutoencoder()
    ae_model.to(device)
    print(f'[DEBUG] Using device: {device}')
    optimizer = torch.optim.Adam(ae_model.parameters())#, lr=lr)
    loss_criterion = torch.nn.L1Loss().to(device)

    # Training the model
    try:
        ae_model.load_state_dict(torch.load('ae_model.pth'))
        print(f'[DEBUG] Model loaded from file')
    except FileNotFoundError:
        print(f'[DEBUG] Model file not found, training new model')
        train_losses, test_losses = ae_train(
            ae_model, trainset, testset, optimizer, loss_criterion, epochs)
        torch.save(ae_model.state_dict(), 'ae_model.pth')
        print(f'[DEBUG] Model saved to file')
        
        plt.figure()
        plt.plot(list(range(1, epochs+1)), train_losses, label='Train Loss')
        plt.plot(list(range(1, epochs+1)), test_losses, label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    # Evaluating the model - Reconstructing a sample of images from the test set
    ae_model.eval()
    reconstructs = None
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(testset):
            inputs = inputs.to(device)
            reconstructs = ae_model(inputs)
            break # Visualizing from a single batch of images

    # Visualizing the reconstructed images
    fig, axs = plt.subplots(2, 8, figsize=(16, 4))
    axs[0, 0].set_ylabel('Original', rotation=0, labelpad=25)
    axs[1, 0].set_ylabel('Reconstructed', rotation=0, labelpad=40)
    for i in range(8):
        # Visualization
        axs[0, i].imshow(inputs[i].squeeze().cpu().numpy(), cmap='gray')
        axs[1, i].imshow(reconstructs[i].squeeze().cpu().numpy(), cmap='gray')
        # Removing axes ticks, they are insignificant
        axs[0, i].get_xaxis().set_ticks([])
        axs[0, i].get_yaxis().set_ticks([])
        axs[1, i].get_xaxis().set_ticks([])
        axs[1, i].get_yaxis().set_ticks([])
    plt.show()

    return ae_model

def q2_ae_classifier(trainset, testset, encoder):
    """
    Experimenting with a simple MNIST convolutional autoencoder.
    The experiment visualizes the training/test losses and the reconstructed images.
    
    :param trainset: The training set.
    :param testset: The testing set.
    :param encoder: The pretrained autoencoder model.
    """
    # Hyperparameters
    lr = 0.01
    epochs = 10

    model = models.ConvEncoderClassifier(encoder)
    model.to(device)
    print(f'[DEBUG] Using device: {device}')
    # Only training the classifier part of the model - The encoder is pretrained
    optimizer = torch.optim.Adam(model.fc.parameters())#, lr=lr)
    loss_criterion = torch.nn.CrossEntropyLoss().to(device)

    # Training the model
    try:
        model.load_state_dict(torch.load('encoder_classifier_q2.pth'))
        print(f'[DEBUG] Model loaded from file')
    except FileNotFoundError:
        print(f'[DEBUG] Model file not found, training new model')
        train_losses, train_accuracy, test_losses, test_accuracy = encoder_classifier_train(
            model, trainset, testset, optimizer, loss_criterion, epochs)
        torch.save(model.state_dict(), 'encoder_classifier_q2.pth')
        print(f'[DEBUG] Model saved to file')
        
        plt.figure()
        plt.title('Encoder-Classifier Loss')
        plt.plot(list(range(1, epochs+1)), train_losses, label='Train Loss')
        plt.plot(list(range(1, epochs+1)), test_losses, label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        plt.figure()
        plt.title('Encoder-Classifier Accuracy')
        plt.plot(list(range(1, epochs+1)), train_accuracy, label='Train Accuracy')
        plt.plot(list(range(1, epochs+1)), test_accuracy, label='Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()
    
    return model

def main():
    # Loading the MNIST dataset (first run may take a little while more
    train, test = load_mnist_data(batch_size=32)

    ### Q1 - Training & evaluating the a MNIST AE model for reconstruction
    ae_model = q1_ae_reconstruction(train, test)
    
    ### Q2 - Training & evaluating a MNIST classifier on the latent space of the AE
    classifier = q2_ae_classifier(train, test, ae_model.encoder)
    #print(encoder_classifier_evaluate(classifier, test, torch.nn.CrossEntropyLoss()))

if __name__ == "__main__":
    main()