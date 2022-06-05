import os
import numpy as np

from load import show_one_sample


if __name__ == "__main__":
    path = '../Data/seq/oracle_source_seq/oracle_source_seq.npz'
    data = np.load(path, allow_pickle = True)

    for sub_mode in ('train', 'test'):
        path = os.path.join('../Data/draw/', sub_mode)
        if not os.path.exists(path):
            os.makedirs(path)
        idx = 0
        for sample in data[sub_mode]:
            show_one_sample(sample, os.path.join(path, '%d.jpg' % idx))
            print('%d.jpg' % idx)
            idx += 1
