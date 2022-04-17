import torch
import torch.nn as nn
import torch.nn.functional as F


def str2bool(string):
    return string.lower() == 'true'


def init_parameter(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


def descent_lr(lr, optimizer, ind_epoch, interval):
    lr = lr * (0.1 ** (ind_epoch // interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('learning rate:', lr)
    print()


def optimize(model, optimizer, train_loader, lr, ind_epoch, interval, record_iter, device):
    model.train()
    print('Training:')

    correct = 0
    count = 0
    error = []

    descent_lr(lr, optimizer, ind_epoch, interval)

    for ind_iter, data in enumerate(train_loader):
        inputs = data[0].to(device)
        targets = data[1].to(device)

        outputs = model(inputs)
        logits = F.log_softmax(outputs)
        loss = F.nll_loss(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        count += inputs.shape[0]
        _, pred = logits.max(dim = 1)
        correct += pred.eq(targets).sum().item()

        if (ind_iter + 1) % record_iter == 0:
            print('epoch:', ind_epoch + 1)
            print('iteration:', ind_iter + 1)
            print('number of samples:', count)
            print('number of correct samples:', correct)
            print('training accuracy: %.5f' % (correct / count))
            print()

            error.append(1 - correct / count)

            count = 0
            correct = 0

    return error


def evaluate(model, data_loader, device):
    model.eval()
    print('Testing:')

    count = 0
    correct_t1 = 0
    correct_t5 = 0

    for _, data in enumerate(data_loader):
        inputs = data[0].to(device)
        targets = data[1].to(device)

        logits = model(inputs)

        count += inputs.shape[0]
        _, pred_t1 = logits.max(dim=1)
        _, pred_t5 = torch.topk(logits, k=5, dim=1)
        correct_t1 += pred_t1.eq(targets).sum().item()
        correct_t5 += pred_t5.eq(torch.unsqueeze(targets, 1).repeat(1, 5)).sum().item()

    print('number of samples:', count)
    print('number of correct samples:', correct_t1)
    print('testing accuracy:', correct_t1 / count)
    print('top 5 testing accuracy:', correct_t5 / count)
    print()

    torch.cuda.empty_cache()
    return 1 - correct_t1 / count


def save_status(model, optimizer, path):
    save_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(save_dict, path)


def load_status(model, optimizer, path):
    load_dict = torch.load(path)
    model.load_state_dict(load_dict['model'])
    optimizer.load_state_dict(load_dict['optimizer'])
