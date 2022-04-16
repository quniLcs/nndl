import argparse
import warnings

import torch
import torch.nn as nn

from slbi_toolbox import SLBI_ToolBox
from slbi_toolbox_Adam import SLBI_ToolBox_Adam

from load import load
from lenet import LeNet
from vgg import VGG_BN
from util import str2bool, init_parameter, optimize, evaluate, save_status

warnings.filterwarnings('ignore')

# python main.py --Adam False
# python main.py --Adam True
parser = argparse.ArgumentParser()
parser.add_argument("--use_cuda", default = True, type = str2bool)
parser.add_argument("--parallel", default = False, type = str2bool)
parser.add_argument("--dataset", default = 'MNIST', type = str)
parser.add_argument("--batch_size", default = 128, type = int)
parser.add_argument("--shuffle", default = True, type = str2bool)
parser.add_argument("--Adam", default = True, type = str2bool)
parser.add_argument("--lr", default = 1e-1, type = float)
parser.add_argument("--kappa", default = 1, type = int)
parser.add_argument("--mu", default = 20, type = int)
parser.add_argument("--epoch", default = 60, type = int)
parser.add_argument("--interval", default = 20, type = int)
parser.add_argument("--record_iter", default = 50, type = int)
args = parser.parse_args()

torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()
device = torch.device("cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")

train_loader = load(dataset = args.dataset, train = True, batch_size = args.batch_size, shuffle = args.shuffle)
test_loader = load(dataset = args.dataset, train = False, batch_size = args.batch_size, shuffle = args.shuffle)

if args.dataset == 'MNIST':
    model = LeNet().to(device)
elif args.dataset == 'CIFAR-10':
    model = VGG_BN().to(device)
else:
    model = None

init_parameter(model)

if args.parallel:
    model = nn.DataParallel(model)

name_layer = []
name_weight = []
for name, parameter in model.named_parameters():
    name_layer.append(name)
    if len(parameter.shape) == 2 or len(parameter.shape) == 4:
        name_weight.append(name)

if args.Adam:
    optimizer = SLBI_ToolBox_Adam(model.parameters(), lr = args.lr, kappa = args.kappa, mu = args.mu, weight_decay = 0)
else:
    optimizer = SLBI_ToolBox(model.parameters(), lr = args.lr, kappa = args.kappa, mu = args.mu, weight_decay = 0)

optimizer.assign_name(name_layer)
optimizer.initialize_slbi(name_weight)

print('number of steps:', args.epoch * len(train_loader))
print('number of steps per epoch:', len(train_loader))
print()

error_train = []
error_test = []

for ind_epoch in range(args.epoch):
    error_train.append(optimize(model, optimizer, train_loader,
                                args.lr, ind_epoch, args.interval, args.record_iter, device))
    optimizer.update_prune_order(ind_epoch)
    error_test.append(evaluate(model, test_loader, device))

save_status(model, optimizer, 'lenet.pth')
torch.save(error_train, 'error_train.pth')
torch.save(error_test, 'error_test.pth')
