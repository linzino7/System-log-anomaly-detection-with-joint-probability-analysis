import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet



class ConvEncoder(BaseNet):
    
    def __init__(self, encoded_space_dim, rep_dim=32):
        super().__init__()

        self.rep_dim = rep_dim
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2), # in_channels, out_channels, kernel_size,
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2),
            nn.ReLU(True)
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(32 *7 *7 , 128),   # chnnel * Hout * Wout from laset conv2d output 32*32=32*3*3    64= 32*7*7
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x


class ConvDecoder(BaseNet):
    
    def __init__(self, encoded_space_dim ):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 32*7*7),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(32, 7, 7))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2,output_padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, output_padding=1)
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x



class Conv_Autoencoder(BaseNet):
    '''
    encoded_space_dim= rep_dim

    '''
    def __init__(self, encoded_space_dim, rep_dim=32, bias=False):
        super().__init__()

        self.rep_dim = rep_dim
        self.encoder = ConvEncoder(encoded_space_dim, rep_dim)
        self.decoder = ConvDecoder(encoded_space_dim)
        

    def forward(self, x):

        out = self.encoder(x)
        out = self.decoder(out)

        return out
