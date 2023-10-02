import torch

x_t = 3.0
lr = 0.1

for it in range(100) :
    x = torch.tensor(x_t, requires_grad=True)
    y = 0.1 * (x**4) + (x**3) + 2 * (x**2) - 5 * x + 2

    print("x : %f y : %f"%(x_t, y))

    y.backward()

    x_t = x_t - lr * x.grad