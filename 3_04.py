import torch
import pickle
import matplotlib.pyplot as plt

broken_image = torch.FloatTensor(pickle.load(open('../3-min-pytorch-master/03-파이토치로_구현하는_ANN/broken_image_t.p',
                                                  'rb'), encoding='latin1'))

plt.imshow(broken_image.view(100, 100))
plt.show()

def weird_function(x, n_iter=5) :
    h = x
    filt = torch.tensor([-1./3, 1./3, -1./3])
    for i in range(n_iter) :
        zero_tensor = torch.tensor([1.0*0])
        h_l = torch.cat((zero_tensor, h[:-1]), 0)
        h_r = torch.cat((h[1:], zero_tensor), 0)
        h = filt[0] * h + filt[2] * h_l + filt[1] * h_r
        if i % 2 == 0 :
            h = torch.cat((h[h.shape[0]//2:], h[:h.shape[0]//2]), 0)
    return h

def distance_loss(hypothesis, broken_image) :
    return torch.dist(hypothesis, broken_image)

random_tensor = torch.randn(10000, dtype = torch.float)

lr = 0.8

for i in range(0, 20000) :
    random_tensor.requires_grad_(True)
    hypothesis = weird_function(random_tensor)
    loss = distance_loss(hypothesis, broken_image)
    loss.backward()

    with torch.no_grad() :
        random_tensor = random_tensor - lr * random_tensor.grad

    if i % 1000 == 0 :
            print('Loss at {} = {}'.format(i, loss.item()))

plt.imshow(random_tensor.view(100, 100).data)
plt.show()
