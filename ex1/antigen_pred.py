import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

NEGATIVE_SAMPLE_PATH = "neg_A0201.txt"
POSITIVE_SAMPLE_PATH = "pos_A0201.txt"

AMINO_ACIDS = ['Y', 'C', 'Q', 'W', 'I', 'D', 'A', 'E', 'K', 'N', 'L', 'G', 'S',
               'H', 'M', 'V', 'T', 'R', 'P', 'F']
AMINO_ACIDS_DICT = {acid: i for i, acid in enumerate(AMINO_ACIDS)}

def preprocess_samples(data):
    """
    Coding each amino acid as a one-hot vector - per its index in the AMINO_ACIDS list
    and its position in the peptide sequence.
    """
    samples = np.zeros(shape=(len(data), len(AMINO_ACIDS) * len(data[0])))

    for idx, line in enumerate(data):
        line = line.strip()
        for acid_idx, acid in enumerate(line):
            samples[idx, AMINO_ACIDS_DICT[acid] + (acid_idx * len(AMINO_ACIDS))] = 1

    return samples

def create_base_nn(input_dim):
    model = nn.Sequential(
        nn.Linear(input_dim, 10), nn.ReLU(), # Input layer
        nn.Linear(10, 10), nn.ReLU(), # Hidden layer 1
        nn.Linear(10, 1) # Output layer
    )
    return model

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

    

if __name__ == "__main__":
    main()