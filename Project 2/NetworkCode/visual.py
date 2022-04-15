import torch


def loss_landscape(losses):
    losses = torch.tensor(losses)
    min_curve = torch.min(losses, dim = 0).values.view(-1)
    max_curve = torch.max(losses, dim = 0).values.view(-1)

    return min_curve, max_curve


def get_dist(parameters):
    num_epoch = len(parameters)
    num_batch = len(parameters[0])

    dists = []

    parameter_last = torch.zeros(parameters[0][0].shape)
    for ind_epoch in range(num_epoch):
        for ind_batch in range(num_batch):
            parameter_current = parameters[ind_epoch][ind_batch]
            dists.append(torch.norm(parameter_current - parameter_last, p = 2))
            parameter_last = parameter_current

    return dists[1:]


def grad_pred(grads):
    dists = []
    for ind in range(len(grads)):
        dists.append(get_dist(grads[ind]))

    min_curve, max_curve = loss_landscape(dists)
    return min_curve, max_curve


def beta_smooth(parameters, grads):
    dists_parameter = []
    for ind in range(len(parameters)):
        dists_parameter.append(get_dist(parameters[ind]))
    dists_parameter = torch.tensor(dists_parameter)

    dists_grad = []
    for ind in range(len(grads)):
        dists_grad.append(get_dist(grads[ind]))
    dists_grad = torch.tensor(dists_grad)

    _, max_curve = loss_landscape(dists_grad / dists_parameter)
    return max_curve
