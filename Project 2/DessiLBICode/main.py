import os
import argparse
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from slbi_toolbox import SLBI_ToolBox

from load import load
from lenet import LeNet
from utils import str2bool, descent_lr, evaluate_batch, save_model_and_optimizer

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("--use_cuda", default = True, type = str2bool)
parser.add_argument("--parallel", default = False, type = str2bool)
parser.add_argument("--gpu_num", default = '0', type = str)
parser.add_argument("--lr", default = 1e-1, type = float)
parser.add_argument("--kappa", default = 1, type = int)
parser.add_argument("--mu", default = 20, type = int)
parser.add_argument("--dataset", default = 'MNIST', type = str)
parser.add_argument("--batch_size", default = 128, type = int)
parser.add_argument("--shuffle", default = True, type = str2bool)
parser.add_argument("--epoch", default = 60, type = int)
parser.add_argument("--interval", default = 20, type = int)
args = parser.parse_args()


torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()
device = torch.device("cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")

train_loader = load(dataset = args.dataset, train = True, batch_size = args.batch_size, shuffle = args.shuffle)
test_loader = load(dataset = args.dataset, train = False, batch_size = 64, shuffle = False)

model = LeNet().to(device)

if args.parallel:
    model = nn.DataParallel(model)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num

name_layer = []
name_weight = []
for name, parameter in model.named_parameters():
    name_layer.append(name)
    if len(parameter.shape) == 2 or len(parameter.shape) == 4:
        name_weight.append(name)

optimizer = SLBI_ToolBox(model.parameters(), lr = args.lr, kappa = args.kappa, mu = args.mu, weight_decay = 0)
optimizer.assign_name(name_layer)
optimizer.initialize_slbi(name_weight)

print('number of steps:', args.epoch * len(train_loader))
print('number of steps per epoch:', len(train_loader))

for ind_epoch in range(args.epoch):
    model.train()
    #
    descent_lr(args.lr, ind_epoch, optimizer, args.interval)

    losses = 0
    correct = 0
    count = 0

    for ind_iter, data in enumerate(train_loader):
        inputs = data[0].to(device)
        targets = data[1].to(device)

        logits = model(inputs)
        loss = F.nll_loss(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses += loss.item()
        count += inputs.shape[0]
        _, pred = logits.max(dim = 1)
        correct += pred.eq(targets).sum().item()

        if (ind_iter + 1) % 50 == 0:
            print('epoch: ', ind_epoch + 1)
            print('iteration: ', ind_iter + 1)
            print('loss: %.5f' % (losses / 50))
            print('number of samples: ', count)
            print('number of correct samples: ', correct)
            print('training accuracy: %.5f' % (correct / count))
            print()

            losses = 0
            correct = 0
            count = 0

        if ind_iter == 251:
            break

    optimizer.update_prune_order(ind_epoch)
    evaluate_batch(model, test_loader, device)

save_model_and_optimizer(model, optimizer, 'lenet.pth')
