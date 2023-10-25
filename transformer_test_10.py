
def make_pad_mask(self query, key, pad_idx=1):
    # query: (n_batch, query_seq_len)
    # key: (n_batch, key_seq_len)
    query_seq_len, key_seq_len = query.size(1), key.size(1)

    key_mask = key.ne(pad_idx).unsqueeze(1).unsqueeze(2)  # (n_batch, 1, 1, key_seq_len)
    key_mask = key_mask.repeat(1, 1, query_seq_len, 1)    # (n_batch, 1, query_seq_len, key_seq_len)

    query_mask = query.ne(pad_idx).unsqueeze(1).unsqueeze(3)  # (n_batch, 1, query_seq_len, 1)
    query_mask = query_mask.repeat(1, 1, 1, key_seq_len)  # (n_batch, 1, query_seq_len, key_seq_len)

    mask = key_mask & query_mask
    mask.requires_grad = False
    return mask