import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet

class IndentityBlock(nn.Module):
    def __init__(self, in_channel, f, filters):
        super(IndentityBlock,self).__init__()
        F1, F2, F3 = filters
        self.stage = nn.Sequential(
            nn.Conv2d(in_channel,F1,1,stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F1),
            nn.ReLU(True),
            nn.Conv2d(F1,F2,f,stride=1, padding=True, bias=False),
            nn.BatchNorm2d(F2),
            nn.ReLU(True),
            nn.Conv2d(F2,F3,1,stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F3),
        )
        self.relu_1 = nn.ReLU(True)
        
    def forward(self, X):
        X_shortcut = X
        X = self.stage(X)
        X = X + X_shortcut
        X = self.relu_1(X)
        return X

class RestConvEncoder_MLP(BaseNet):
    
    def __init__(self, encoded_space_dim, conv_dim , unflattened_size , rep_dim=32, topN=64):
        super().__init__()

        self.rep_dim = rep_dim
        self.conv_dim = conv_dim
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2), # in_channels, out_channels, kernel_size,
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            IndentityBlock(8, 3, [2, 2, 8]),
            IndentityBlock(8, 3, [2, 2, 8]),
            nn.Conv2d(8, 16, 3, stride=2),
            #nn.BatchNorm2d(16),
            nn.ReLU(True),
            IndentityBlock(16, 3, [4, 4, 16]),
            IndentityBlock(16, 3, [4, 4, 16]),
            nn.Conv2d(16, 32, 3, stride=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2,padding=1),
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(conv_dim+topN, 128),   # chnnel * Hout * Wout from laset conv2d output #64 = 32*7*7 #32 =32*3*3
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )
        
        
    def forward(self, graph, top64):
        x = self.encoder_cnn(graph)
        x = self.flatten(x)
        x = torch.cat((x,top64),1) # cat 
        x = self.encoder_lin(x)
        return x


class ConvDecoder_MLP(BaseNet):
    
    def __init__(self, encoded_space_dim,conv_dim,unflattened_size, topN=64):
        super().__init__()
        self.conv_dim = conv_dim
        self.unflattened_size = unflattened_size
        self.topN = topN
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, conv_dim+topN),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=unflattened_size)

        self.decoder_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ConvTranspose2d(32, 16, 2, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            IndentityBlock(16, 3, [4, 4, 16]),
            IndentityBlock(16, 3, [4, 4, 16]),
            nn.ConvTranspose2d(16, 8, 2, stride=2,output_padding=0),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            IndentityBlock(8, 3, [2, 2, 8]),
            IndentityBlock(8, 3, [2, 2, 8]),
            nn.ConvTranspose2d(8, 1, 2, stride=2, output_padding=0)
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)

        idxs = list(range(self.conv_dim))
        indices_64 = torch.tensor(idxs,device='cuda')
        graph= torch.index_select(x, 1, indices_64)

        idxs = list(range(self.conv_dim, self.conv_dim+self.topN))
        indices = torch.tensor(idxs, device='cuda')
        top64 = torch.index_select(x, 1, indices)

        x = self.unflatten(graph)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x, top64



class RestConv_Autoencoder_MLP(BaseNet):
    '''
    encoded_space_dim= rep_dim

    '''
    def __init__(self, encoded_space_dim, conv_dim, unflattened_size, rep_dim=32, topN=64, bias=False):
        super().__init__()

        self.rep_dim = rep_dim
        print('AE',topN)
        self.encoder = RestConvEncoder_MLP(encoded_space_dim, conv_dim, unflattened_size, rep_dim, topN)
        self.decoder = ConvDecoder_MLP(encoded_space_dim, conv_dim,unflattened_size, topN)
        

    def forward(self, graph, top64):

        out = self.encoder(graph,top64)
        out, top64 = self.decoder(out)

        return out, top64
