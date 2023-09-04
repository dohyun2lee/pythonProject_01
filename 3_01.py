import torch

x = torch.tensor([[1,2,3,1],[4,5,6,1],[7,8,9,1],[10,11,12,1]])

print(x)
print("size", x.size())
print("shape", x.shape)
print("rank", x.ndimension())

x = torch.unsqueeze(x, 1)

print("unsqueeze!\n", x)
print("size", x.size())
print("shape", x.shape)
print("rank", x.ndimension())

x = torch.squeeze(x)

print("squeeze!\n", x)
print("size", x.size())
print("shape", x.shape)
print("rank", x.ndimension())

x = x.view(9)

try :
    x = x.view(2,4)
except Exception as error :
    print(error)

print(x)
print("size", x.size())
print("shape", x.shape)
print("rank", x.ndimension())