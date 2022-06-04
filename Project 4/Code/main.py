import argparse
import time
import os
import logging

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18

from load import data_loader_builder
from utils import train


def str_to_bool(value):
    return value.lower() == 'true'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epoch", default = 200, type = int)
    parser.add_argument("--batch_size", default = 8, type = int)
    parser.add_argument("--num_workers", default = 4, type = int)
    parser.add_argument("--mode", default = 'train', type = str)         # pretrain, train
    parser.add_argument("--form", default = 'img', type = str)           # img, seq
    parser.add_argument("--augment", default = False, type = str_to_bool) # True, False
    parser.add_argument("--shot", default = 1, type = int)               # 1, 3, 5
    parser.add_argument("--lr", default = 0.0001, type = float)          # 0.001, 0.0001
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.time = time.strftime('%Y%m%d%H%M')
    args.file = '%s_%d_%s_%s' % (args.form, args.shot, 'augment' if args.augment else 'baseline', args.time)

    assert args.mode in ('pretrain', 'train')
    assert args.form in ('img', 'seq')
    assert args.shot in (1, 3, 5)

    runs = 'runs/%s_%s' % (args.mode, args.file)
    if not os.path.exists(runs):
        os.makedirs(runs)
    writer = SummaryWriter(runs)

    logpath = 'logs/'
    if not os.path.exists(logpath):
        os.makedirs(logpath)
    logging.basicConfig(level = logging.INFO)
    logfile = 'logs/%s_%s.log' % (args.mode, args.file)
    loghandler = logging.FileHandler(logfile, mode = 'w')
    logformatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d]: %(message)s")
    loghandler.setFormatter(logformatter)
    logger = logging.getLogger(__name__)
    logger.addHandler(loghandler)

    torch.backends.cudnn.benchmark = True

    logger.info("Loading the data...")
    if args.mode == 'pretrain':
        data_loader = data_loader_builder(args, mode = 'pre-train', form = args.form, shuffle = True)
    else: # args.mode == 'train':
        if args.augment:
            train_loader = data_loader_builder(args, mode = 'train', form = 'seq', shuffle = True)
        else:
            train_loader = data_loader_builder(args, mode = 'train', form = 'img', shuffle=True)
        train_img_loader = data_loader_builder(args, mode = 'train', form = 'img', shuffle = False)
        test_img_loader = data_loader_builder(args, mode = 'test', form = 'img', shuffle=False)
    logger.info("Finish the data loading.")

    if args.mode == 'pretrain':
        pass
    else: # args.mode = 'train':
        classifier = resnet18(pretrained = True).to(args.device)
        classifier.fc = torch.nn.Linear(512, 200)
        if args.augment:
            pass
        else:
            augmentor = None
        train(classifier, augmentor, train_loader, train_img_loader, test_img_loader, writer, logger, args)

    writer.flush()
    writer.close()