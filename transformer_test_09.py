import torch.nn as nn

class Transformer(nn.Module):

    ...

    def encode(self, src, src_mask):
        out = self.encoder(src, src_mask)
        return out


    def forward(self, src, tgt, src_mask):
        encoder_out = self.encode(src, src_mask)
        y = self.decode(tgt, encoder_out)
        return y

    ...