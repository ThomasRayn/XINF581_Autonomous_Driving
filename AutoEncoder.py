import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

# This file provide the basic classes for autoencoder and variational autoencoder implementation

# Credit to https://medium.com/dataseries/convolutional-autoencoder-in-pytorch-on-mnist-dataset-d65145c132ac that proposes an autoencoder architecture
class MyCropFunction:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        
    def __call__(self, x):
        return transforms.functional.crop(x, 0, 0, self.height, self.width)
    

class Encoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, stride=2, padding=0),
            nn.ReLU(True)
        )
        
        self.flatten = nn.Flatten(start_dim=1)
        
        self.encoder_lin = nn.Sequential(
            nn.Linear(10 * 10 * 256, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim),
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        
        return x
    
    def transform_data(self, x):
        transform = torchvision.transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.Grayscale(),
            MyCropFunction(height=83, width=83),
            transforms.ToTensor(),
        ])
        
        return transform(x)

class Decoder(nn.Module):
    
    def __init__(self, encoded_space_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 10 * 10 * 256)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(256, 10, 10))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 3, stride=2, padding=1)
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        return x
    


# Credit to the TP_8 of the course XINF-581 that inspires a variational autoencoder architecture

class VAE(nn.Module):
    def __init__(self, z_dim=20):
        super(VAE, self).__init__()
        
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, stride=2, padding=0),
            nn.ReLU(True),
            nn.Flatten(start_dim=1),
            nn.Linear(10 * 10 * 256, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, z_dim + z_dim)
        )
        
        self.decoder_cnn = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 10 * 10 * 256),
            nn.Unflatten(dim=1, unflattened_size=(256, 10, 10)),
            nn.ConvTranspose2d(256, 128, 3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 3, stride=2, padding=1)
        )
        
    def encode(self, x):
        h = self.encoder_cnn(x)
        return torch.tensor_split(h, 2, axis=1)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_cnn(z)
        return torch.sigmoid(h)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var
