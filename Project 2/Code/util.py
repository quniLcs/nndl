import numpy as np


def get_num_parameters(model):
    num_parameters = 0
    for parameter in model.parameters():
        num_parameters += np.prod(parameter.shape)
    return num_parameters
