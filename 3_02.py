import torch

w = torch.randn(5,3, dtype=torch.float)
x = torch.tensor([[1.0,2.0], [3.0,4.0], [5.0,6.0]])
b = torch.randn(5,2, dtype=torch.float)

print("w :", w)
print("x :", x)
print("b :", b)
print("w size :", w.size())
print("x size :", x.size())
print("b size :", b.size())
print()

wx = torch.mm(w, x)

print("wx size :", wx.size())
print("wx :", wx)
print()

plus_result = wx + b

print("plus_result size :", plus_result.size())
print("plus_result :", plus_result)
