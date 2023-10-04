import copy
import torch.nn as nn

class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, d_model, h, qkv_fc, out_fc):
        super(MultiHeadAttentionLayer, self).__init__()
        self.d_model = d_model
        self.h = h
        self.q_fc = copy.deepcopy(qkv_fc) # (d_embed, d_model)
        self.k_fc = copy.deepcopy(qkv_fc) # (d_embed, d_model)
        self.v_fc = copy.deepcopy(qkv_fc) # (d_embed, d_model)
        self.out_fc = out_fc              # (d_model, d_embed)

        ...