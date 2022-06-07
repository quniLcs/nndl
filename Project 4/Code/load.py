import os
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import matplotlib.pyplot as plt


def strokes_to_lines(strokes):
    x = 0
    y = 0
    line = []
    lines = []

    for i in range(len(strokes)):
        x += strokes[i, 0]
        y += strokes[i, 1]
        line.append([x, y])
        if strokes[i, 2] == 1:
            lines.append(line)
            line = []

    return lines


def show_one_sample(sample, file = None):
    lines = strokes_to_lines(sample)

    plt.figure(figsize = (50, 50), dpi = 1)
    for i in range(len(lines)):
        x = [line[0] for line in lines[i]]
        y = [line[1] for line in lines[i]]
        plt.plot(x, y, 'k', linewidth = 100)

    plt.xticks([])
    plt.yticks([])
    plt.gca().invert_yaxis()

    if file:
        plt.savefig(file)
    else:
        plt.show()


def img_preprocesser(sample):
    return 1 - sample / 255


def seq_preprocesser(before):
    length = min(before.shape[0], 300)

    after = np.zeros((300, 4))
    after[:length, :3] = np.stack((before[:length, 0] / 49,
                                  before[:length, 1] / 49,
                                  before[:length, 2]), axis = 1)
    after[length:, 3] = 1

    return after


def pretrain_img_dataset_builder(split = 'train'):
    dataset = []

    path = os.path.join('../Data/draw/', split)
    for sample in os.listdir(path):
        dataset.append(img_preprocesser(read_image(os.path.join(path, sample))[0]))
    """
    path = '../Data/img/oracle_source_img'
    sub_path = 'bnu_xxt_hard'
    cur_sub_path = os.path.join(path, sub_path)
    for sample in os.listdir(cur_sub_path):
        dataset.append(img_preprocesser(read_image(os.path.join(cur_sub_path, sample))))

    for sub_path in ('gbk_bronze_lst_seal', 'oracle_54081', 'other_font'):
        cur_path = os.path.join(path, sub_path)
        for idx in os.listdir(cur_path):
            cur_sub_path = os.path.join(cur_path, idx)
            for sample in os.listdir(cur_sub_path):
                dataset.append(img_preprocesser(read_image(os.path.join(cur_sub_path, sample))))
    """
    return dataset


def pretrain_seq_dataset_builder(split = 'train'):
    dataset = []

    path = '../Data/seq/oracle_source_seq/oracle_source_seq.npz'
    data = np.load(path, allow_pickle = True)

    path = os.path.join('../Data/draw/', split)
    for idx in tqdm(range(len(data[split]))):
        dataset.append((torch.from_numpy(seq_preprocesser(data[split][idx])),
                        img_preprocesser(read_image(os.path.join(path, '%d.jpg' % idx))[0])))

    return dataset


def train_test_img_dataset_builder(split = 'train', shot = 1):
    dataset = []

    path = '../Data/img/oracle_200_%d_shot/%s' % (shot, split)
    for idx in range(200):
        sub_path = os.path.join(path, '%d' % idx)
        for sample in os.listdir(sub_path):
            dataset.append((img_preprocesser(read_image(os.path.join(sub_path, sample))), idx))

    return dataset


def train_test_seq_dataset_builder(split = 'train', shot = 1):
    dataset = []

    path = '../Data/seq/oracle_200_%d_shot' % shot
    for idx in range(200):
        data = np.load(os.path.join(path, '%d.npz' % idx), allow_pickle = True)
        for sample in data[split]:
            dataset.append((torch.from_numpy(seq_preprocesser(sample)), idx))

    return dataset


class OracleDataset(Dataset):
    def __init__(self, mode = 'pretrain', split = 'train', form = 'img', shot = 1):
        assert form in ('img', 'seq')
        assert mode in ('pretrain', 'train')
        assert split in ('train', 'test')
        assert shot in (1, 3, 5)

        if mode == 'pretrain':
            if form == 'img':
                self.dataset = pretrain_img_dataset_builder(split = split)
            else: # form == 'seq'
                self.dataset = pretrain_seq_dataset_builder(split = split)

        else: # mode == 'train'
            if form == 'img':
                self.dataset = train_test_img_dataset_builder(split = split, shot = shot)
            else: # form == 'seq'
                self.dataset = train_test_seq_dataset_builder(split = split, shot = shot)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def data_loader_builder(args, split, form, shuffle):
    dataset = OracleDataset(mode = args.mode, split = split, form = form, shot = args.shot)
    data_loader = DataLoader(dataset,
                             batch_size = args.batch_size,
                             shuffle = shuffle,
                             num_workers = args.num_workers)
    return data_loader


if __name__ == "__main__":
    path = '../Data/seq/oracle_source_seq/oracle_source_seq.npz'
    data = np.load(path, allow_pickle = True)

    data_train = data['train']
    data_test = data['test']

    print(len(data_train))
    print(len(data_test))

    show_one_sample(data_train[0])
