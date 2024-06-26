# 3rd parties
import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

# 1st parties
import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_mnist_data(subset=0, batch_size=32, target_dir='./data'):
    """
    Downloading and loading the MNIST dataset

    :param subset: The number of samples to use from the training set. 0 means the whole training set.
    :param batch_size: The batch size for the data loader
    :param target_dir: The directory where the data will be downloaded
    :return: The train and test data loaders
    """
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(
                                                    mean=[0.5], std=[0.5])])
    train_set = torchvision.datasets.MNIST(
        root=target_dir, train=True, download=True, transform=transform)
    if subset > 0:
        indices = torch.arange(100)
        subset = torch.utils.data.Subset(train_set, indices)
    else:
        subset = train_set
    train_loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)
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

    ae_model = models.ConvAutoencoder().to(device)
    print(f'[DEBUG] Using device: {device}')
    optimizer = torch.optim.Adam(ae_model.parameters())#, lr=lr)
    loss_criterion = torch.nn.L1Loss().to(device)

    # Training the model
    try:
        ae_model.load_state_dict(torch.load('ae_model_q1.pth'))
        print(f'[DEBUG] Model loaded from file')
    except FileNotFoundError:
        print(f'[DEBUG] Model file not found, training new model')
        train_losses, test_losses = ae_train(
            ae_model, trainset, testset, optimizer, loss_criterion, epochs)
        torch.save(train_losses, 'train_losses_q1.pth')
        torch.save(test_losses, 'test_losses_q1.pth')
        torch.save(ae_model.state_dict(), 'ae_model_q1.pth')
        print(f'[DEBUG] Model saved to file')
        
        plt.figure()
        plt.plot(list(range(1, epochs+1)), train_losses, label='Train Loss')
        plt.plot(list(range(1, epochs+1)), test_losses, label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('q1_autoencoder_loss.pdf')

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
    plt.savefig('q1_reconstructed_images.pdf')

    return ae_model

def q2_ae_classifier(trainset, testset, encoder=None, title='q2', epochs=10):
    """
    Experimenting with a simple MNIST convolutional autoencoder.
    The experiment visualizes the training/test losses and the reconstructed images.
    
    :param trainset: The training set.
    :param testset: The testing set.
    :param encoder: The pretrained autoencoder model for fine-tuning.
    :param title: The title of the experiment.
    :return: The trained encoder-classifier model.
    """
    # Hyperparameters
    lr = 0.01
    epochs = epochs

    # Creating the model with the pretrained encoder or a new encoder
    model = models.ConvEncoderClassifier(encoder)
    model.to(device)
    print(f'[DEBUG] Using device: {device}')
    learnable_params = model.parameters()
    optimizer = torch.optim.Adam(learnable_params)#, lr=lr)
    loss_criterion = torch.nn.CrossEntropyLoss().to(device)

    # Training the model
    try:
        model.load_state_dict(torch.load(f'encoder_classifier_{title}.pth'))
        print(f'[DEBUG] Model loaded from file')
    except FileNotFoundError:
        print(f'[DEBUG] Model file not found, training new model')
        train_losses, train_accuracy, test_losses, test_accuracy = encoder_classifier_train(
            model, trainset, testset, optimizer, loss_criterion, epochs)
        torch.save(train_losses, f'train_losses_{title}.pth')
        torch.save(train_accuracy, f'train_accuracy_{title}.pth')
        torch.save(test_losses, f'test_losses_{title}.pth')
        torch.save(test_accuracy, f'test_accuracy_{title}.pth')
        torch.save(model.state_dict(), f'encoder_classifier_{title}.pth')
        print(f'[DEBUG] Model saved to file')
        
        plt.figure()
        plt.title('Encoder-Classifier Loss')
        plt.plot(list(range(1, epochs+1)), train_losses, label='Train Loss')
        plt.plot(list(range(1, epochs+1)), test_losses, label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'{title}_encoder_classifier_loss.pdf')

        plt.figure()
        plt.title('Encoder-Classifier Accuracy')
        plt.plot(list(range(1, epochs+1)), train_accuracy, label='Train Accuracy')
        plt.plot(list(range(1, epochs+1)), test_accuracy, label='Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(f'{title}_encoder_classifier_accuracy.pdf')
    
    return model

def q2_visualize_misclassifications(model, testset):
    """
    Visualizing the misclassified images by the model.

    :param model: The model to visualize the misclassifications.
    :param testset: The testing set.
    """
    model.eval()
    misclass_inputs = []
    misclass_input_labels = []
    misclass_outputs = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(testset):
            labels = labels.to(device)
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            misclassified = outputs.argmax(1) != labels
            misclass_inputs += list(inputs[misclassified])
            misclass_input_labels += list(labels[misclassified])
            misclass_outputs += list(outputs.argmax(1)[misclassified])

    # Visualizing (some) misclassified images
    fig, axs = plt.subplots(4, 8, figsize=(16, 8))
    for i in range(min(len(misclass_inputs), 8)):
        for j in range(4):
            current_index = ((i+1)*j) + i
            # Visualization
            axs[j, i].imshow(misclass_inputs[current_index].squeeze().cpu().numpy(), cmap='gray')
            axs[j, i].set_title(
                f'True: {misclass_input_labels[current_index]}\nPredicted: {misclass_outputs[current_index]}', 
                fontdict={'fontsize': 6})
            # Removing axes ticks, they are insignificant
            axs[j, i].get_xaxis().set_ticks([])
            axs[j, i].get_yaxis().set_ticks([])
    plt.savefig('q2_misclassified_images.pdf')

def q3_ae_decoder_finetune(trainset, testset, pretrained_encoder):
    """
    """
    # Hyperparameters
    lr = 0.01
    epochs = 10

    # Training only a decoder from scratch with the pretrained encoder
    ae_model = models.ConvAutoencoder(pretrained_encoder).to(device)
    optimizer = torch.optim.Adam(ae_model.parameters())#, lr=lr)
    loss_criterion = torch.nn.L1Loss().to(device)

    # Training the model
    try:
        ae_model.load_state_dict(torch.load('ae_model_q3.pth'))
        print(f'[DEBUG] Model loaded from file')
    except FileNotFoundError:
        print(f'[DEBUG] Model file not found, training new model')
        train_losses, test_losses = ae_train(
            ae_model, trainset, testset, optimizer, loss_criterion, epochs)
        torch.save(train_losses, 'train_losses_q3.pth')
        torch.save(test_losses, 'test_losses_q3.pth')
        torch.save(ae_model.state_dict(), 'ae_model_q3.pth')
        print(f'[DEBUG] Model saved to file')
        
        train_loss_q1 = torch.load('train_losses_q1.pth')
        test_loss_q1 = torch.load('test_losses_q1.pth')

        plt.figure()
        plt.plot(list(range(1, epochs+1)), train_loss_q1, label='Train Loss Q1')
        plt.plot(list(range(1, epochs+1)), test_loss_q1, label='Test Loss Q1')
        plt.plot(list(range(1, epochs+1)), train_losses, label='Train Loss Q3')
        plt.plot(list(range(1, epochs+1)), test_losses, label='Test Loss Q3')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('q3_autoencoder_loss.pdf')

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
    plt.savefig('q3_reconstructed_images.pdf')

    return ae_model

def q4_ae_small_subset_classifier(trainset, testset):
    """
    """
    # Hyperparameters
    epochs = 20

    return q2_ae_classifier(trainset, testset, title='q4', epochs=epochs)

def main():
    # Loading the MNIST dataset (first run may take a little while more
    train, test = load_mnist_data(batch_size=32)

    ### Q1 - Training & evaluating the a MNIST AE model for reconstruction
    print('[DEBUG] Q1')
    ae_model = q1_ae_reconstruction(train, test)
    
    ### Q2 - Training & evaluating a MNIST classifier on the latent space of the AE
    print('[DEBUG] Q2')
    classifier = q2_ae_classifier(train, test)
    q2_visualize_misclassifications(classifier, test)
    #classifier = q2_ae_classifier(train, test, ae_model.encoder, fine_tune=True)

    ### Q3 - Fine-tuning Decoder for better reconstruction
    print('[DEBUG] Q3')
    ae_model = q3_ae_decoder_finetune(train, test, classifier.encoder)

    ### Q4 - Training on a small subset of the MNIST dataset
    print('[DEBUG] Q4')
    train_small, test_small = load_mnist_data(subset=100, batch_size=32)
    # Using the same classifier as in Q2
    classifier = q4_ae_small_subset_classifier(train_small, test_small)

if __name__ == "__main__":
    main()