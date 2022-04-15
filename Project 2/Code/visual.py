import numpy as np


def loss_landscape(losses):
    num_epoch = len(losses[0])
    num_batch = len(losses[0][0])

    min_curve = [np.inf for _ in range(num_epoch * num_batch)]
    max_curve = [-np.inf for _ in range(num_epoch * num_batch)]

    for ind in range(len(losses)):
        for ind_epoch in range(num_epoch):
            for ind_batch in range(num_batch):
                if losses[ind][ind_epoch][ind_batch] < min_curve[ind_epoch * num_batch + ind_batch]:
                    min_curve[ind_epoch * num_batch + ind_batch] = losses[ind][ind_epoch][ind_batch]
                elif losses[ind][ind_epoch][ind_batch] > max_curve[ind_epoch * num_batch + ind_batch]:
                    max_curve[ind_epoch * num_batch + ind_batch] = losses[ind][ind_epoch][ind_batch]

    return min_curve, max_curve
