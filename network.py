import torch

class NeuralNet(torch.nn.Module) :
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.fc1 = torch.nn.Linear(self.input_size, 1)
        self.act1 = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)

        return x
