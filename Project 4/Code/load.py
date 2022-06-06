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
        plt.savefig(file, bbox_inches = 'tight', pad_inches = 0)
    else:
        plt.show()


def img_preprocesser(sample):
    return (sample - 127) / 128


def seq_preprocesser(before):
    length = min(before.shape[0], 300)
    after = np.zeros((300, 3))
    after[:length, :] = np.stack((before[:length, 0] / 49,
                                  before[:length, 1] / 49,
                                  before[:length, 2]), axis = 1)
    after[length:, 2] = 2
    return after


def pretrain_img_dataset_builder():
    dataset = []

    path = '../Data/img/oracle_source_img'
    sub_path = 'bnu_xxt_hard'
    cur_sub_path = os.path.join(path, sub_path)
    for sample in os.listdir(cur_sub_path):
        dataset.append({'img': img_preprocesser(read_image(os.path.join(cur_sub_path, sample)))})

    for sub_path in ('gbk_bronze_lst_seal', 'oracle_54081', 'other_font'):
        cur_path = os.path.join(path, sub_path)
        for idx in os.listdir(cur_path):
            cur_sub_path = os.path.join(cur_path, idx)
            for sample in os.listdir(cur_sub_path):
                dataset.append(img_preprocesser(read_image(os.path.join(cur_sub_path, sample))))

    return dataset


def pretrain_seq_dataset_builder():
    dataset = []

    path = '../Data/seq/oracle_source_seq/oracle_source_seq.npz'
    data = np.load(path, allow_pickle = True)
    for sub_mode in ('train', 'test'):
        cur_sub_path = os.path.join('../Data/draw/', sub_mode)
        idx = 0
        for sample in data[sub_mode]:
            dataset.append((torch.from_numpy(seq_preprocesser(sample)),
                            img_preprocesser(read_image(os.path.join(cur_sub_path, '%d.jpg' % idx)))))
            idx += 1

    return dataset


def train_test_img_dataset_builder(mode = 'train', shot = 1):
    dataset = []

    path = '../Data/img/oracle_200_%d_shot/%s' % (shot, mode)
    for idx in range(200):
        cur_sub_path = os.path.join(path, '%d' % idx)
        for sample in os.listdir(cur_sub_path):
            dataset.append((img_preprocesser(read_image(os.path.join(cur_sub_path, sample))), idx))

    return dataset


def train_test_seq_dataset_builder(mode = 'train', shot = 1):
    dataset = []

    path = '../Data/seq/oracle_200_%d_shot' % shot
    for idx in range(200):
        data = np.load(os.path.join(path, '%d.npz' % idx), allow_pickle = True)
        for sample in data[mode]:
            dataset.append((torch.from_numpy(seq_preprocesser(sample)), idx))

    return dataset


class OracleDataset(Dataset):
    def __init__(self, mode = 'pre-train', form = 'img', shot = 1):
        assert form in ('img', 'seq')
        assert mode in ('pre-train', 'train', 'test')
        assert shot in (1, 3, 5)
        self.mode = mode
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
        return self.dataset[idx]

def data_loader_builder(args, mode, form, shuffle):
    dataset = OracleDataset(mode = mode, form = form, shot = args.shot)
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