import torch
import cv2
import numpy as np

size = 256
num_training = 10000
lr = 10

random_tensor = torch.randn(size=(256,256,3), dtype=torch.float)
random_tensor = torch.clip(random_tensor, min=-1.0, max=1.0)

#vis_rt = (random_tensor + 1) / 2.0 * 255.0
#vis_rt = vis_rt.numpy()
# cv2.imshow('rt', vis_rt.astype(np.uint8))
# cv2.waitKey(-1)

target = cv2.imread('../img/cat1.jpg')
target = cv2.resize(target, (size, size))
cv2.imshow('target', target)
#cv2.waitKey(-1)

target = (target / 255.0) * 2 - 1
target = torch.from_numpy(target)

for it in range(num_training) :
    random_tensor.requires_grad_(True)
    loss = torch.mean((target - random_tensor) ** 2)
    loss.backward()

    with torch.no_grad() :
        random_tensor = random_tensor - lr * random_tensor.grad

    print("it : %d, loos val : %.5f" %(it, loss))

    vis_rt = (random_tensor + 1) / 2.0 * 255.0
    vis_rt = vis_rt.numpy()
    cv2.imshow('rt', vis_rt.astype(np.uint8))
    cv2.waitKey(1)