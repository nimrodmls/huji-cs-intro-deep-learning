import torch
import torch.nn as nn

class ConvEncoder(nn.Module):
    """
    Convolutional encoder for the MNIST dataset.
    """

    def __init__(self, in_channels, latent_dim):
        super(ConvEncoder, self).__init__()

        # Convolutional part
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

        # Bottleneck part
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            # 7x7 is the dimension of the image after the two convolutions with stride=2
            nn.Linear(32 * 7 * 7, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, latent_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.bottleneck(self.conv(x))
    
class ConvDecoder(nn.Module):
    """
    Convolutional decoder for the MNIST dataset.
    """

    def __init__(self, in_channels, latent_dim):
        super(ConvDecoder, self).__init__()

        # Bottleneck part
        self.bottleneck = nn.Sequential(
            nn.Linear(latent_dim, 32 * 7 * 7),
            nn.ReLU(True),
            nn.Unflatten(1, (32, 7, 7))
        )

        # Convolutional part
        self.conv = nn.Sequential(
            # Note that the output padding is necessary to restore the correct output size
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.conv(self.bottleneck(x))
    
class ConvAutoencoder(nn.Module):
    """
    Convolutional autoencoder for the MNIST dataset.
    """

    IN_CHANNELS = 1 # The MNIST dataset has 1 channel (greyscale images)
    LATENT_DIM = 12 # The dimension of the latent space

    def __init__(self, encoder=None, decoder=None):
        super(ConvAutoencoder, self).__init__()

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = ConvEncoder(ConvAutoencoder.IN_CHANNELS, ConvAutoencoder.LATENT_DIM)

        if decoder is not None:
            self.decoder = decoder
        else:
            self.decoder = ConvDecoder(ConvAutoencoder.IN_CHANNELS, ConvAutoencoder.LATENT_DIM)

    def forward(self, x):
        return self.decoder(self.encoder(x))
    
class ConvEncoderClassifier(nn.Module):
    """
    Convolutional encoder for the MNIST dataset.
    """
    OUT_DIM = 10 # MNIST has 10 classes
    HIDDEN_DIM = 128 # Hidden layer dimension

    def __init__(self, encoder=None, classifier=None):
        super(ConvEncoderClassifier, self).__init__()

        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = ConvEncoder(ConvAutoencoder.IN_CHANNELS, ConvAutoencoder.LATENT_DIM)
        # Fully connected layer for classification
        if classifier is not None:
            self.fc = classifier
        else:
            self.fc = nn.Sequential(
                nn.Linear(ConvAutoencoder.LATENT_DIM, ConvEncoderClassifier.HIDDEN_DIM),
                nn.BatchNorm1d(ConvEncoderClassifier.HIDDEN_DIM),
                nn.ReLU(True),
                nn.Linear(ConvEncoderClassifier.HIDDEN_DIM, ConvEncoderClassifier.OUT_DIM),
                nn.Softmax(1)
            )

    def forward(self, x):
        return self.fc(self.encoder(x))