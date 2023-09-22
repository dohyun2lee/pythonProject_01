import copy
import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self, encoder_block, n_layer):  # n_layer: Encoder Block의 개수
        super(Encoder, self).__init__()
        self.layers = []
        for i in range(n_layer):
            self.layers.append(copy.deepcopy(encoder_block))


    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

class EncoderBlock(nn.Module):

    def __init__(self, self_attention, position_ff):
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention
        self.position_ff = position_ff


    def forward(self, x):
        out = x
        out = self.self_attention(out)
        out = self.position_ff(out)
        return out