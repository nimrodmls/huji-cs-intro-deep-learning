import torch
import torch.nn as nn

class ConvEncoder(nn.Module):
    """
    Convolutional encoder for the MNIST dataset.
    """
    IN_CHANNELS = 1 # The MNIST dataset has 1 channel (greyscale images)
    LATENT_DIM = 12 # The dimension of the latent space

    def __init__(self):
        super(ConvEncoder, self).__init__()

        # TODO: Experiment with pooling layers
        self.encoder = nn.Sequential(
            nn.Conv2d(ConvEncoder.IN_CHANNELS, 6, kernel_size=3, stride=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(6, ConvEncoder.LATENT_DIM, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.encoder(x)
    
class ConvDecoder(nn.Module):
    """
    Convolutional decoder for the MNIST dataset.
    """

    def __init__(self):
        super(ConvDecoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(ConvEncoder.LATENT_DIM, 6, kernel_size=3, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(6, ConvEncoder.IN_CHANNELS, kernel_size=3, stride=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)
    
class ConvAutoencoder(nn.Module):
    """
    Convolutional autoencoder for the MNIST dataset.
    """

    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        self.encoder = ConvEncoder()
        self.decoder = ConvDecoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))
    
class ConvEncoderMLP(nn.Module):
    """
    Convolutional encoder for the MNIST dataset.
    """
    OUT_DIM = 10 # MNIST has 10 classes
    HIDDEN_DIM = 20 # Hidden layer dimension

    def __init__(self):
        super(ConvEncoderMLP, self).__init__()

        self.encoder = ConvEncoder()
        self.fc = nn.Sequential(
            nn.Linear(ConvEncoder.LATENT_DIM, ConvEncoderMLP.HIDDEN_DIM),
            nn.ReLU(True),
            nn.Linear(ConvEncoderMLP.HIDDEN_DIM, ConvEncoderMLP.OUT_DIM),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(self.encoder(x))