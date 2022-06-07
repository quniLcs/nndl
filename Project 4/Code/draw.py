import warnings
import os
from tqdm import tqdm
import numpy as np

from load import show_one_sample


if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    path = '../Data/seq/oracle_source_seq/oracle_source_seq.npz'
    data = np.load(path, allow_pickle = True)

    for split in ('train', 'test'):
        path = os.path.join('../Data/draw/', split)
        if not os.path.exists(path):
            os.makedirs(path)

        idx = 0
        for sample in tqdm(data[split]):
            show_one_sample(sample, os.path.join(path, '%d.jpg' % idx))
            idx += 1