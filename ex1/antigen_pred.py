import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

NEGATIVE_SAMPLE_PATH = "neg_A0201.txt"
POSITIVE_SAMPLE_PATH = "pos_A0201.txt"
# The ratio of negative samples to positive samples is 8:1,
# this will come in handy when creating the training & testing datasets
# and weighing the loss function
NEGATIVE_SAMPLES_PER_POSITIVE = 8

# The ratio of the training set to the testing set is 10:90
TRAIN_SET_RATIO = 0.1

# The amino acids in the peptide sequence
AMINO_ACIDS = ['Y', 'C', 'Q', 'W', 'I', 'D', 'A', 'E', 'K', 'N', 'L', 'G', 'S',
               'H', 'M', 'V', 'T', 'R', 'P', 'F']
AMINO_ACIDS_DICT = {acid: i for i, acid in enumerate(AMINO_ACIDS)}

def one_hot_to_peptide(one_hot):
    """
    """
    peptide = ""
    for i in range(0, len(one_hot), len(AMINO_ACIDS)):
        max_idx = np.argmax(one_hot[i : i + len(AMINO_ACIDS)])
        peptide += AMINO_ACIDS[max_idx]
    return peptide

def preprocess_samples(data):
    """
    Coding each amino acid as a one-hot vector - per its index in the AMINO_ACIDS list
    and its position in the peptide sequence.
    """
    samples = np.zeros(shape=(len(data), len(AMINO_ACIDS) * len(data[0].strip())))

    for idx, line in enumerate(data):
        line = line.strip()
        for acid_idx, acid in enumerate(line):
            samples[idx, AMINO_ACIDS_DICT[acid] + (acid_idx * len(AMINO_ACIDS))] = 1

    return samples

def generate_datasets(negative_samples, positive_samples):
    """
    Creating training & testing datasets from the negative & positive samples.
    """
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

    train_set_positive_samples_count = train_set_size // NEGATIVE_SAMPLES_PER_POSITIVE
    train_set_negative_samples_count = train_set_size - train_set_positive_samples_count
    test_set_positive_samples_count = len(positive_samples) - train_set_positive_samples_count
    test_set_negative_samples_count = len(negative_samples) - train_set_negative_samples_count

    train_positive_indices = np.random.choice(
        positive_samples_remainder, train_set_positive_samples_count, replace=False)
    test_positive_indices = np.random.choice(
        positive_samples_remainder, test_set_positive_samples_count, replace=False)
    train_negative_indices = np.random.choice(
        negative_samples_remainder, train_set_negative_samples_count, replace=False)
    test_negative_indices = np.random.choice(
        negative_samples_remainder, test_set_negative_samples_count, replace=False)

    train_set[:train_set_positive_samples_count] = positive_samples[train_positive_indices]
    train_labels[:train_set_positive_samples_count] = 1
    train_set[train_set_positive_samples_count:] = negative_samples[train_negative_indices]

    test_set[:test_set_positive_samples_count] = positive_samples[test_positive_indices]
    test_labels[:test_set_positive_samples_count] = 1
    test_set[test_set_positive_samples_count:] = negative_samples[test_negative_indices]
    
    train_perm = np.random.permutation(len(train_set))
    test_perm = np.random.permutation(len(test_set))

    return train_set[train_perm], train_labels[train_perm], \
          test_set[test_perm], test_labels[test_perm]

def create_base_nn(input_dim):
    model = nn.Sequential(
        nn.Linear(input_dim, input_dim), nn.ReLU(), # Input layer
        # Creating 2 hidden layers with the same dimension as the input layer
        # as specified in the exercise description
        nn.Linear(input_dim, input_dim), nn.ReLU(), # Hidden layer 1
        nn.Linear(input_dim, input_dim), nn.ReLU(), # Hidden layer 2
        nn.Linear(10, 1) # Output layer
    )
    return model

def train_model(model, train_data, epochs=100, lr=0.01, batch_size=32):
    # We should consider the imbalance of the dataset when setting the weights on
    # the CrossEntropyLoss function

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model.to(device)
    # print('[DEBUG] Using device:', device)

    # trainset = torch.utils.data.TensorDataset(torch.tensor(train_data[['long', 'lat']].values).float(), torch.tensor(train_data['country'].values).long())
    # valset = torch.utils.data.TensorDataset(torch.tensor(val_data[['long', 'lat']].values).float(), torch.tensor(val_data['country'].values).long())
    # testset = torch.utils.data.TensorDataset(torch.tensor(test_data[['long', 'lat']].values).float(), torch.tensor(test_data['country'].values).long())

    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    # valloader = torch.utils.data.DataLoader(valset, batch_size=1024, shuffle=False, num_workers=0)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False, num_workers=0)

    # criterion = nn.CrossEntropyLoss().to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # for ep in range(epochs):
    #     # TRAINING
    #     model.train()
    #     pred_correct = 0
    #     ep_loss = 0.
    #     for i, (inputs, labels) in enumerate(tqdm(trainloader)):
    #         # perform a training iteration
    #         inputs = inputs.to(device)
    #         labels = labels.to(device)

    #         optimizer.zero_grad()
    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         if callback:
    #             callback(model, ep)
    #         optimizer.step()

    #         pred_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
    #         ep_loss += loss.item()

    #     train_accs.append(pred_correct / len(trainset))
    #     train_losses.append(ep_loss / len(trainloader))
    pass

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
    train_set, train_labels, test_set, test_labels = generate_datasets(negative_samples, positive_samples)
    print('bp')

    # Creating the NN based on the dimension of the one-hot encoded samples
    # (we do this on the negative samples, but in reality it's the same dimension for both)
    #model = create_base_nn(input_dim=negative_samples.shape[1])
    #train_model(model)

if __name__ == "__main__":
    main()