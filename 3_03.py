import torch

w = torch.tensor(1.0, requires_grad=True)

a = w * 3

l = a ** 2

l.backward()

print("w :", w)
print("a :", a)
print("l :", l)
print("l을 w로 미분한 값 :", format(w.grad))