import torch.nn as nn

class Transformer(nn.Module):

    def __init__(self, encoder, decoder):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder


    def encode(self, x):
        out = self.encoder(x)
        return out


    def decode(self, z, c):
        out = self.decode(z, c)
        return out


    def forward(self, x, z):
        c = self.encode(x)
        y = self.decode(z, c)
        return y

