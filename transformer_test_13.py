class ResidualConnectionLayer(nn.Module):

    def __init__(self):
        super(ResidualConnectionLayer, self).__init__()


    def forward(self, x, sub_layer):
        out = x
        out = sub_layer(out)
        out = out + x
        return out
