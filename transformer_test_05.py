import torch.nn as nn

class MultiHeadAttentionLayer(nn.Module):
    ...


def forward(self, *args, query, key, value, mask=None):
    # query, key, value: (n_batch, seq_len, d_embed)
    # mask: (n_batch, seq_len, seq_len)
    # return value: (n_batch, h, seq_len, d_k)
    n_batch = query.size(0)

    def transform(x, fc):  # (n_batch, seq_len, d_embed)
        out = fc(x)  # (n_batch, seq_len, d_model)
        out = out.view(n_batch, -1, self.h, self.d_model // self.h)  # (n_batch, seq_len, h, d_k)
        out = out.transpose(1, 2)  # (n_batch, h, seq_len, d_k)
        return out

    query = transform(query, self.q_fc)  # (n_batch, h, seq_len, d_k)
    key = transform(key, self.k_fc)  # (n_batch, h, seq_len, d_k)
    value = transform(value, self.v_fc)  # (n_batch, h, seq_len, d_k)

    out = self.calculate_attention(query, key, value, mask)  # (n_batch, h, seq_len, d_k)
    out = out.transpose(1, 2)  # (n_batch, seq_len, h, d_k)
    out = out.contiguous().view(n_batch, -1, self.d_model)  # (n_batch, seq_len, d_model)
    out = self.out_fc(out)  # (n_batch, seq_len, d_embed)
    return out

