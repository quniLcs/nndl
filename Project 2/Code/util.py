import random
import numpy as np
import torch


def set_random_seeds(seed = 0, device = 'cpu'):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device != 'cpu':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_num_parameters(model):
    num_parameters = 0
    for parameter in model.parameters():
        num_parameters += np.prod(parameter.shape)
    return num_parameters
