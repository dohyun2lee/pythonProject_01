def make_src_mask(self, src):
    pad_mask = self.make_pad_mask(src, src)
    return pad_mask