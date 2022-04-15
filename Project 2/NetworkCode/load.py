import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader


def load(root = '../Data/', train = True, batch_size = 128, shuffle = True, num_workers = 4):
    torch.manual_seed(0)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])])

    dataset = datasets.CIFAR10(root = root, train = train, download = True, transform = transform)
    loader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)

    return loader


if __name__ == '__main__':
    train_loader = load()
    for x, y in train_loader:
        print(x[0])
        print(y[0])

        print(x[0].shape)
        print(x[0].max())
        print(x[0].min())

        img = np.transpose(x[0], [1, 2, 0])
        plt.imshow(img * 0.5 + 0.5)
        plt.show()
        break
