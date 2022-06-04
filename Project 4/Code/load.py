import os
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


def show_one_sample(sample):
    lines = strokes_to_lines(sample)

    for i in range(len(lines)):
        x = [line[0] for line in lines[i]]
        y = [line[1] for line in lines[i]]
        plt.plot(x, y, 'k')

    plt.xticks([])
    plt.yticks([])
    plt.gca().invert_yaxis()
    plt.show()


def img_preprocesser(sample):
    return (sample - 127) / 128


def seq_preprocesser(sample):
    return np.stack((sample[:, 0] / 49, sample[:, 1] / 49, sample[:, 2] - 0.5), axis = 1)


def pretrain_img_dataset_builder():
    dataset = []

    path = '../Data/img/oracle_source_img'
    sub_path = 'bnu_xxt_hard'
    cur_path = os.path.join(path, sub_path)
    for sample in os.listdir(cur_path):
        dataset.append({'img': 1 - read_image(os.path.join(cur_path, sample)) / 255})

    for sub_path in ('gbk_bronze_lst_seal', 'oracle_54081', 'other_font'):
        cur_path = os.path.join(path, sub_path)
        for character in os.listdir(cur_path):
            cur_sub_path = os.path.join(cur_path, character)
            for sample in os.listdir(cur_sub_path):
                dataset.append({'img': img_preprocesser(read_image(os.path.join(cur_sub_path, sample)))})

    return dataset


def pretrain_seq_dataset_builder():
    dataset = []

    path = '../Data/seq/oracle_source_seq/oracle_source_seq.npz'
    data = np.load(path, allow_pickle = True)
    for sub_mode in ('train', 'test'):
        for sample in data[sub_mode]:
            dataset.append({'seq': torch.from_numpy(seq_preprocesser(sample))})

    return dataset


def train_test_img_dataset_builder(mode = 'train', shot = 1):
    dataset = []

    path = '../Data/img/char_to_idx.txt'
    with open(path, 'r', encoding = 'utf-8') as file:
        char_to_idx = file.read()

    path = '../Data/img/oracle_200_%d_shot/%s' % (shot, mode)
    for character in os.listdir(path):
        cur_sub_path = os.path.join(path, character)
        for sample in os.listdir(cur_sub_path):
            dataset.append({'img': img_preprocesser(read_image(os.path.join(cur_sub_path, sample))),
                            'char': character, 'idx': char_to_idx.index(character)})

    return dataset


def train_test_seq_dataset_builder(mode = 'train', shot = 1):
    dataset = []

    path = '../Data/seq/char_to_idx.txt'
    with open(path, 'r', encoding='utf-8') as file:
        char_to_idx = file.read()

    path = '../Data/seq/oracle_200_%d_shot' % shot
    for idx in range(200):
        data = np.load(os.path.join(path, '%d.npz' % idx), allow_pickle = True)
        for sample in data[mode]:
            dataset.append({'seq': torch.from_numpy(seq_preprocesser(sample)),
                            'char': char_to_idx[idx], 'idx': idx})

    return dataset


class OracleDataset(Dataset):
    def __init__(self, mode = 'pre-train', form = 'img', shot = 1):
        assert form in ('img', 'seq')
        assert mode in ('pre-train', 'train', 'test')
        assert shot in (1, 3, 5)
        self.form = form

        if mode == 'pre-train':
            if form == 'img':
                self.dataset = pretrain_img_dataset_builder()
            else: # form == 'seq'
                self.dataset = pretrain_seq_dataset_builder()

        else: # mode == 'train' or mode == 'test'
            if form == 'img':
                self.dataset = train_test_img_dataset_builder(mode = mode, shot = shot)
            else: # form == 'seq'
                self.dataset = train_test_seq_dataset_builder(mode = mode, shot = shot)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx][self.form], self.dataset[idx]['idx']


def data_loader_builder(args, mode, form, shuffle):
    dataset = OracleDataset(mode = mode, form = form, shot = args.shot)
    data_loader = DataLoader(dataset,
                             batch_size = args.batch_size,
                             shuffle = shuffle,
                             num_workers = args.num_workers)
    return data_loader


if __name__ == "__main__":
    path = '../Data/oracle_source/oracle_source_seq/oracle_source_seq.npz'
    data = np.load(path, allow_pickle = True)

    data_train = data['train']
    data_test = data['test']

    show_one_sample(data_train[0])