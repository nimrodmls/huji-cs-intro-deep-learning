import torch
import torch.nn as nn

class ConvEncoder(nn.Module):
    """
    Convolutional encoder for the MNIST dataset.
    """

    def __init__(self, in_channels, latent_dim, mid_dims):
        super(ConvEncoder, self).__init__()

        # TODO: Experiment with pooling layers

        #conv_layer = lambda i, o: nn.Conv2d(i, o, kernel_size=3, stride=2, padding=1)
        #act_layer = lambda i, o: nn.ReLU(True)

        # Convolutional part
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            #*[layer(mid_dims[i], mid_dims[i+1]) for i in range(len(mid_dims)-1) for layer in [conv_layer, act_layer]],
            nn.Conv2d(6, 12, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True)
        )

        # Linear part
        self.linear = nn.Sequential(
            nn.Flatten(),
            # 7x7 is the dimension of the image after the two convolutions
            nn.Linear(12 * 7 * 7, latent_dim)
        )

    def forward(self, x):
        return self.linear(self.conv(x))
    
class ConvDecoder(nn.Module):
    """
    Convolutional decoder for the MNIST dataset.
    """

    def __init__(self, in_channels, latent_dim):
        super(ConvDecoder, self).__init__()

        # Linear part
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 12 * 7 * 7),
            nn.Unflatten(1, (12, 7, 7))
        )

        # Convolutional part
        self.conv = nn.Sequential(
            # Note that the output padding is necessary to restore the correct output size
            nn.ConvTranspose2d(12, 6, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(6, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(self.linear(x))
    
class ConvAutoencoder(nn.Module):
    """
    Convolutional autoencoder for the MNIST dataset.
    """

    IN_CHANNELS = 1 # The MNIST dataset has 1 channel (greyscale images)
    LATENT_DIM = 12 # The dimension of the latent space

    def __init__(self):
        super(ConvAutoencoder, self).__init__()

        self.encoder = ConvEncoder(ConvAutoencoder.IN_CHANNELS, ConvAutoencoder.LATENT_DIM)
        self.decoder = ConvDecoder(ConvAutoencoder.IN_CHANNELS, ConvAutoencoder.LATENT_DIM)

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