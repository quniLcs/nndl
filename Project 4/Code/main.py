import argparse
import time
import os
import logging

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18

from load import data_loader_builder
from deberta import OrcDeBERTa
from transformers import ViTMAEConfig, ViTMAEForPreTraining
from utils import load_status, pretrain, train


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epoch", default = 200, type = int)
    parser.add_argument("--batch_size", default = 8, type = int)
    parser.add_argument("--num_workers", default = 4, type = int)
    parser.add_argument("--mode", default = 'train', type = str)  # pretrain, train
    parser.add_argument("--form", default = 'img', type = str)    # baseline, tradition, cutout, mixup, cutmix, img, seq
    parser.add_argument("--shot", default = 1, type = int)        # 1, 3, 5
    parser.add_argument("--lr", default = 0.0001, type = float)   # 0.001, 0.0001
    args = parser.parse_args()
    # python main.py --mode train --form baseline --shot 1 --lr 0.0001
    # python main.py --mode train --form tradition --shot 1 --lr 0.0001
    # python main.py --mode train --form cutout --shot 1 --lr 0.0001
    # python main.py --mode train --form mixup --shot 1 --lr 0.0001
    # python main.py --mode train --form cutmix --shot 1 --lr 0.0001
    # python main.py --mode pretrain --form img --lr 0.001
    # python main.py --mode train --form img --shot 1 --lr 0.0001
    # python main.py --mode pretrain --form seq --lr 0.001
    # python main.py --mode train --form seq --shot 1 --lr 0.0001

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.time = time.strftime('%Y%m%d%H%M')
    args.file = '%s_%d_%s' % (args.form, args.shot, args.time)

    assert args.mode in ('pretrain', 'train')
    assert args.form in ('img', 'seq', 'baseline', 'tradition', 'cutout', 'mixup', 'cutmix')
    assert args.shot in (1, 3, 5)

    results = 'results'
    if not os.path.exists(results):
        os.makedirs(results)

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

    if args.mode == 'pretrain':
        logger.info('Load the data...')
        train_loader = data_loader_builder(args, split = 'train', form = args.form, shuffle = True)
        test_loader = data_loader_builder(args, split = 'test', form = args.form, shuffle = False)
        logger.info('Finish data loading.')

        if args.form == 'img':
            config = ViTMAEConfig(image_size = 50, patch_size = 5, mask_ratio = 0.15,
                                  hidden_size = 128, decoder_hidden_size = 128,
                                  num_hidden_layers = 4, decoder_num_hidden_layers = 4,
                                  num_attention_heads = 4, decoder_num_attention_heads = 4,
                                  intermediate_size = 512, decoder_intermediate_size = 512)
            augmentor = ViTMAEForPreTraining(config).to(args.device)
        else:
            augmentor = OrcDeBERTa().to(args.device)

        pretrain(augmentor, train_loader, test_loader, writer, logger, args)

    else: # args.mode == 'train':
        logger.info('Load the data...')
        if args.form in ('img', 'seq'):
            train_loader = data_loader_builder(args, split = 'train', form = args.form, shuffle = True)
        else:
            train_loader = data_loader_builder(args, split = 'train', form = 'img', shuffle = True)
        train_img_loader = data_loader_builder(args, split = 'train', form = 'img', shuffle = False)
        test_img_loader = data_loader_builder(args, split = 'test', form = 'img', shuffle = False)
        logger.info('Finish data loading.')

        classifier = resnet18(pretrained = True)
        classifier.fc = torch.nn.Linear(512, 200)
        classifier = classifier.to(args.device)

        if args.form in ('img', 'seq'):
            if args.form == 'img':
                config = ViTMAEConfig(image_size = 50, patch_size = 5, mask_ratio = 0.15,
                                      hidden_size = 128, decoder_hidden_size = 128,
                                      num_hidden_layers = 4, decoder_num_hidden_layers = 4,
                                      num_attention_heads = 4, decoder_num_attention_heads = 4,
                                      intermediate_size = 512, decoder_intermediate_size = 512)
                augmentor = ViTMAEForPreTraining(config).to(args.device)
            else:
                augmentor = OrcDeBERTa().to(args.device)
            load_status(augmentor, 'augmentor.pth')
        else:
            augmentor = None

        train(classifier, augmentor, train_loader, train_img_loader, test_img_loader, writer, logger, args)

    writer.flush()
    writer.close()
