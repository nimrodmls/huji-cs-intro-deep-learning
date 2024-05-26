from tqdm import tqdm
import torch
from torch.utils.data import WeightedRandomSampler
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List

# Hyperparameters
EPOCHS = 10
LEARNING_RATE = 0.01
BATCH_SIZE = 32

# Paths for the data files - Change according to the location of the files on your machine
NEGATIVE_SAMPLE_PATH = "neg_A0201.txt"
POSITIVE_SAMPLE_PATH = "pos_A0201.txt"
SPIKE_PROTEIN_PATH = "spike.txt"

# The ratio of the training set to the testing set is 10:90
TRAIN_SET_RATIO = 0.9

# The amino acids in the peptide sequence
AMINO_ACIDS = ['Y', 'C', 'Q', 'W', 'I', 'D', 'A', 'E', 'K', 'N', 'L', 'G', 'S',
               'H', 'M', 'V', 'T', 'R', 'P', 'F']
AMINO_ACIDS_DICT = {acid: i for i, acid in enumerate(AMINO_ACIDS)}

class AntigenPredictor(nn.Module):
    """
    The neural network model for predicting antigen with protein peptides.
    """
    def __init__(self, input_dim):
        super(AntigenPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim), nn.ReLU(), # Input layer
            # Creating 2 hidden layers with the same dimension as the input layer
            # as specified in the exercise description
            nn.Linear(input_dim, input_dim), nn.ReLU(), # Hidden layer 1
            nn.Linear(input_dim, input_dim), nn.ReLU(), # Hidden layer 2
            nn.Linear(input_dim, 1), nn.Sigmoid() # Output layer
        )

    def forward(self, x):
        return self.net(x)

def one_hot_to_peptide(one_hot):
    """
    Utility function for converting a one-hot vector of peptide to a string peptide sequence.
    :param one_hot: The one-hot vector of the peptide
    :return: The peptide sequence
    """
    peptide = ""
    for i in range(0, len(one_hot), len(AMINO_ACIDS)):
        max_idx = np.argmax(one_hot[i : i + len(AMINO_ACIDS)])
        peptide += AMINO_ACIDS[max_idx]
    return peptide

def preprocess_samples(data: List[str]):
    """
    Coding each amino acid as a one-hot vector - per its index in the AMINO_ACIDS list
    and its position in the peptide sequence.
    :param data: The samples to preprocess. Each sample is a peptide sequence.
    :return: The one-hot encoded samples.
    """
    samples = np.zeros(shape=(len(data), len(AMINO_ACIDS) * len(data[0].strip())))

    for idx, line in enumerate(data):
        line = line.strip()
        for acid_idx, acid in enumerate(line):
            samples[idx, AMINO_ACIDS_DICT[acid] + (acid_idx * len(AMINO_ACIDS))] = 1

    return samples


def get_spike_dataset():
    """
    """
    spike_data = None
    with open(SPIKE_PROTEIN_PATH, "r") as f:
        spike_data = "".join(f.read().replace('\n', ''))

    # Extracting all the 9-mer peptides from the spike protein
    spike_protein = [spike_data[i:i+9] for i in range(len(spike_data) - 8)]

    return spike_protein

def generate_datasets(negative_samples, positive_samples, batch_size=32):
    """
    Creating training & testing datasets from the negative & positive samples.
    """
    print("[DEBUG] Generating datasets...")

    # Determining the size of the training & testing sets
    total_samples = len(negative_samples) + len(positive_samples)
    train_set_size = int(np.floor(total_samples * TRAIN_SET_RATIO))
    test_set_size = total_samples - train_set_size

    train_set = np.zeros(shape=(train_set_size, negative_samples.shape[1]))
    train_labels = np.zeros(shape=(train_set_size))
    test_set = np.zeros(shape=(test_set_size, negative_samples.shape[1]))
    test_labels = np.zeros(shape=(test_set_size))

    # Using those ranges to randomly select samples for the training & testing sets
    negative_samples_remainder = list(range(len(negative_samples)))
    positive_samples_remainder = list(range(len(positive_samples)))

    # Calculating the number of positive & negative samples in the training & testing sets
    # as per the globally-defined ratios
    train_set_positive_samples_count = int(np.floor(len(positive_samples) * TRAIN_SET_RATIO))
    train_set_negative_samples_count = train_set_size - train_set_positive_samples_count
    test_set_positive_samples_count = len(positive_samples) - train_set_positive_samples_count
    test_set_negative_samples_count = len(negative_samples) - train_set_negative_samples_count

    # Randomly selecting samples for the training & testing sets
    train_positive_indices = np.random.choice(
        positive_samples_remainder, train_set_positive_samples_count, replace=False)
    test_positive_indices = list(set(positive_samples_remainder) - set(train_positive_indices))

    train_negative_indices = np.random.choice(
        negative_samples_remainder, train_set_negative_samples_count, replace=False)
    test_negative_indices = list(set(negative_samples_remainder) - set(train_negative_indices))

    # Building the training set & its corresponding labels,
    # we insert the positive samples first and then the negative samples
    # the order is later ignored by random sampling
    train_set[:train_set_positive_samples_count] = positive_samples[train_positive_indices]
    train_labels[:train_set_positive_samples_count] = 1
    train_set[train_set_positive_samples_count:] = negative_samples[train_negative_indices]
    # Weighing the samples in the training set according to the imbalance of the dataset
    train_set_weights = np.zeros(shape=(train_set_size))
    train_set_weights[:train_set_positive_samples_count] = 1. / train_set_positive_samples_count
    train_set_weights[train_set_positive_samples_count:] = 1. / train_set_negative_samples_count
    sampler = WeightedRandomSampler(train_set_weights, train_set_size, replacement=True)

    # Building the testing set & its corresponding labels, in the same manner as the training set
    test_set[:test_set_positive_samples_count] = positive_samples[test_positive_indices]
    test_labels[:test_set_positive_samples_count] = 1
    test_set[test_set_positive_samples_count:] = negative_samples[test_negative_indices]
    # Weighted sampler for the test set is not required as it's only evaluated and not trained on
    
    trainset = torch.utils.data.TensorDataset(
        torch.tensor(train_set).float(), torch.tensor(train_labels).long())
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, sampler=sampler)
    
    testset = torch.utils.data.TensorDataset(
        torch.tensor(test_set).float(), torch.tensor(test_labels).long())
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    print("[DEBUG] Dataset generation complete")

    return trainloader, testloader

def train_model(model, trainloader, epochs=10, lr=0.01):
    """
    Training the model on the training set.
    :param model: The model to train.
    :param trainloader: The training set.
    :param epochs: The number of epochs to train the model.
    :param lr: The learning rate for the optimizer.
    :return: The losses and accuracies of the model during training, for each epoch.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f'[DEBUG] Using device: {device}')

    criterion = torch.nn.BCELoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    losses = []
    accuracies = []
    for ep in range(epochs):

        print(f'[TRAIN] Epoch: {ep + 1}')
        model.train()
        pred_correct = 0
        ep_loss = 0.

        for i, (inputs, labels) in enumerate(tqdm(trainloader)):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()

            pred_correct += (outputs.round().squeeze() == labels).sum().item()
            ep_loss += loss.item()

        accuracies.append(pred_correct / len(trainloader.dataset))
        losses.append(ep_loss / len(trainloader))

    print('[TRAIN] Final Loss: ', losses[-1])
    print('[TRAIN] Final Accuracy: ', accuracies[-1])

    return losses, accuracies

def test_model(model, testloader):
    """
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f'[DEBUG] Using device: {device}')

    print('[TEST] Starting evaluation...')
    model.eval()
    with torch.no_grad():
        correct = 0
        total_positive = 0
        correct_positive_preds_cnt = 0
        for i, (inputs, labels) in enumerate(tqdm(testloader)):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            # Rounding the outputs to 0 or 1 to get the predicted labels
            outputs = outputs.round().squeeze()

            # Finding all the true positives corresponding to the positive samples
            correct_positive_preds = outputs.int() & labels.int()
            # Counting the number of correct positive predictions
            correct_positive_preds_cnt += (correct_positive_preds).sum().item()
            # Counting the total number of positive samples
            total_positive += labels.sum().item()
            # Counting the total number of correct predictions
            correct += (outputs == labels).sum().item()

        print(f'[TEST] Evaluation Recall: {correct_positive_preds_cnt / total_positive}')
        print(f'[TEST] Evaluation Accuracy: {correct / len(testloader.dataset)}')

def test_spike(model):
    """
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f'[DEBUG] Using device: {device}')

    print('[SPIKE TEST] Starting evaluation...')
    model.eval()
    spike_onehots = preprocess_samples(get_spike_dataset())

    spike_results = torch.zeros(size=(spike_onehots.shape[0],))
    with torch.no_grad():
        for i, (inputs) in enumerate(tqdm(spike_onehots)):
            samples = torch.zeros(size=(1,inputs.shape[0]), device=device)
            samples[0] = torch.tensor(inputs)
            outputs = model(samples)

            outputs = outputs.squeeze()
            spike_results[i] = outputs.item()

    top3_results = spike_results.topk(3)
    top3_results = {one_hot_to_peptide(spike_onehots[index]): value.item() for index, value in zip(top3_results.indices, top3_results.values)}
    print(f'[SPIKE TEST] Top 3 peptides: {top3_results}')

def plot_sequence(x, y, title, xlabel, ylabel):
    """
    Generic plotting
    """
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.plot(x, y, label=ylabel, color='blue')
    plt.legend()
    plt.show()

def main():
    # Loading all negative & positive samples
    negative_samples = None
    with open(NEGATIVE_SAMPLE_PATH, "r") as f:
        neg_data = f.readlines()
        negative_samples = preprocess_samples(neg_data)
    
    positive_samples = None
    with open(POSITIVE_SAMPLE_PATH, "r") as f:
        pos_data = f.readlines()
        positive_samples = preprocess_samples(pos_data)

    # Creating the training & testing datasets
    trainloader, testloader = \
        generate_datasets(negative_samples, positive_samples, batch_size=BATCH_SIZE)

    # Creating the NN based on the dimension of the one-hot encoded samples
    # (we do this on the negative samples, but in reality it's the same dimension for both)
    model = AntigenPredictor(input_dim=negative_samples.shape[1])
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Trying to load a pre-trained model if it exists
    # IMPORTANT: If you wish to retrain the model, delete the antigen_predictor.pth file
    try:
        model.load_state_dict(torch.load('antigen_predictor.pth'))
        print('[DEBUG] Pre-trained model loaded')
    except FileNotFoundError:
        print('[DEBUG] No pre-trained model found, starting training')
        # No pre-trained model exists, so we proceed to training
        train_losses, train_accuracies = train_model(model, trainloader, epochs=EPOCHS, lr=LEARNING_RATE)
        plot_sequence(
            range(len(train_losses)), train_losses, 'Training Loss per Epoch', 'Epoch', 'Loss')
        plot_sequence(
            range(len(train_accuracies)), train_accuracies, 'Training Accuracy per Epoch', 'Epoch', 'Accuracy')
        torch.save(model.state_dict(), 'antigen_predictor.pth')
    
    # Testing
    test_model(model, testloader)

    # Testing with the spike set
    test_spike(model)

if __name__ == "__main__":
    main()