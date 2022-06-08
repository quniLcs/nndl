import numpy as np
import torch


def cutout(inputs, length = 16, device = torch.device('cpu')):
    n, c, h, w = inputs.shape

    for i in range(n):
        mask = torch.ones(h, w).to(device)

        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - length // 2, 0, h)
        y2 = np.clip(y + length // 2, 0, h)
        x1 = np.clip(x - length // 2, 0, w)
        x2 = np.clip(x + length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.

        mask = mask.repeat(c, 1, 1)
        inputs[i] = inputs[i] * mask

    return


def mixup(inputs, targets, alpha = 1):
    index = torch.randperm(inputs.shape[0])
    lambd = np.random.beta(alpha, alpha)

    inputs_mixup = lambd * inputs + (1 - lambd) * inputs[index]
    targets_mixup = targets[index]

    return inputs_mixup, targets_mixup, lambd, index


def cutmix(inputs, targets, alpha = 1):
    n, c, h, w = inputs.shape

    index = torch.randperm(n)
    lambd = np.random.beta(alpha, alpha)

    r = np.sqrt(1 - lambd)

    y = np.random.randint(h)
    x = np.random.randint(w)

    y1 = np.clip(y - int(h * r) // 2, 0, h)
    y2 = np.clip(y + int(h * r) // 2, 0, h)
    x1 = np.clip(x - int(w * r) // 2, 0, w)
    x2 = np.clip(x + int(w * r) // 2, 0, w)

    lambd = 1 - (x2 - x1) * (y2 - y1) / (h * w)

    inputs_cutmix = inputs.clone()
    inputs_cutmix[:, :, x1: x2, y1: y2] = inputs_cutmix[index, :,  x1: x2,  y1: y2]
    targets_cutmix = targets[index]

    return inputs_cutmix, targets_cutmix, lambd, index