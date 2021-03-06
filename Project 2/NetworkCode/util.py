import random
import numpy as np
import torch
from tqdm import tqdm


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
        num_parameters += torch.prod(parameter.shape)
    return num_parameters


def wrap_tqdm(data_loader, wrap_tqdms, ind_epoch, num_epochs):
    if wrap_tqdms:
        return tqdm(data_loader, unit = 'batch', desc = 'Epoch: %2d/%2d' % (ind_epoch + 1, num_epochs))
    else:
        return data_loader


def optimize_loss(model, optimizer, criterion, train_loader, device, wrap_tqdms, ind_epoch, num_epochs):
    model.train()
    losses = []

    for inputs, labels in wrap_tqdm(train_loader, wrap_tqdms, ind_epoch, num_epochs):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses


def optimize_plus(model, optimizer, criterion, train_loader, device, wrap_tqdms, ind_epoch, num_epochs):
    # the method to get the parameter is referred to:
    # https://blog.csdn.net/weixin_44058333/article/details/92691656
    # the method to get the gradient  is referred to:
    # https://blog.csdn.net/qq_41554005/article/details/119767740
    model.train()
    parameters = []
    grads = []

    for inputs, labels in wrap_tqdm(train_loader, wrap_tqdms, ind_epoch, num_epochs):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        parameter_weight = []
        parameter_bias = []
        grad_weight = []
        grad_bias = []

        for name, parameter_current in model.named_parameters():
            if name == list(model.state_dict())[-2]:
                parameter_weight = parameter_current
                grad_weight = parameter_current.grad
            elif name == list(model.state_dict())[-1]:
                parameter_bias = parameter_current
                grad_bias = parameter_current.grad

        parameter = torch.vstack((torch.transpose(parameter_weight, 0, 1), parameter_bias))
        parameters.append(parameter)

        grad = torch.vstack((torch.transpose(grad_weight, 0, 1), grad_bias))
        grads.append(grad)

    return parameters, grads


def eval_error(model, data_loader, device, wrap_tqdms, ind_epoch, num_epochs):
    model.eval()

    correct = 0
    total = 0

    for inputs, labels in wrap_tqdm(data_loader, wrap_tqdms, ind_epoch, num_epochs):
        with torch.no_grad():
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

        _, pred = outputs.topk(1)
        correct += pred.eq(labels.view(-1, 1)).sum()
        total += inputs.size(0)

    error = 1 - correct / total
    return error


def train(model, optimizer, criterion, train_loader, test_loader, num_epochs = 20, device = 'cpu',
          wrap_tqdms = False, print_errors = False,
          best_model_file = '', losses_file = '',
          train_errors_file = '', test_errors_file = ''):
    set_random_seeds(seed = 0, device = device)
    model.to(device)

    losses = []
    train_errors = []
    test_errors = []
    min_test_error = 1

    for ind_epoch in range(num_epochs):
        loss = optimize_loss(model, optimizer, criterion, train_loader, device, wrap_tqdms, ind_epoch, num_epochs)
        losses.append(loss)

        train_error = eval_error(model, train_loader, device, wrap_tqdms, ind_epoch, num_epochs)
        test_error = eval_error(model, test_loader, device, wrap_tqdms, ind_epoch, num_epochs)

        train_errors.append(train_error)
        test_errors.append(test_error)

        if print_errors:
            print('Epoch: %2d\tTrain Error: %.5f\tTest Error: %.5f' % (ind_epoch + 1, train_error, test_error))

        if best_model_file and test_error < min_test_error:
            min_test_error = test_error
            torch.save(model, best_model_file)

    if losses_file:
        torch.save(losses, losses_file)
    if train_errors_file:
        torch.save(train_errors, train_errors_file)
    if test_errors_file:
        torch.save(test_errors, test_errors_file)

    return losses, train_errors, test_errors


def train_plus(model, optimizer, criterion, train_loader, test_loader, num_epochs = 20, device = 'cpu',
               wrap_tqdms = False, print_errors = False,
               best_model_file = '', parameters_file = '', grads_file = '',
               train_errors_file = '', test_errors_file = ''):
    # the understanding of the paper is referred to: https://zhuanlan.zhihu.com/p/66683061

    set_random_seeds(seed = 0, device = device)
    model.to(device)

    parameters = []
    grads = []
    train_errors = []
    test_errors = []
    min_test_error = 1

    for ind_epoch in range(num_epochs):
        parameter, grad = optimize_plus(model, optimizer, criterion, train_loader, device,
                                        wrap_tqdms, ind_epoch, num_epochs)
        parameters.append(parameter)
        grads.append(grad)

        train_error = eval_error(model, train_loader, device, wrap_tqdms, ind_epoch, num_epochs)
        test_error = eval_error(model, test_loader, device, wrap_tqdms, ind_epoch, num_epochs)

        train_errors.append(train_error)
        test_errors.append(test_error)

        if print_errors:
            print('Epoch: %2d\tTrain Error: %.5f\tTest Error: %.5f' % (ind_epoch + 1, train_error, test_error))

        if best_model_file and test_error < min_test_error:
            min_test_error = test_error
            torch.save(model, best_model_file)

    if parameters_file:
        torch.save(parameters, parameters_file)
    if grads_file:
        torch.save(grads, grads_file)
    if train_errors_file:
        torch.save(train_errors, train_errors_file)
    if test_errors_file:
        torch.save(test_errors, test_errors_file)

    return parameters, grads, train_errors, test_errors
