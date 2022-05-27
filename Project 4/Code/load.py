# load and visualize .npz file

import numpy as np
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

def show_one_sample(strokes):
    lines = strokes_to_lines(strokes)
    for i in range(len(lines)):
        x = [line[0] for line in lines[i]]
        y = [line[1] for line in lines[i]]
        plt.plot(x, y, 'k')

    plt.xticks([])
    plt.yticks([])
    plt.gca().invert_yaxis()
    plt.show()


if __name__ == "__main__":
    path = '../Data/oracle_source/oracle_source_seq/oracle_source_seq.npz'
    data = np.load(path, allow_pickle = True)

    data_train = data['train']
    data_test = data['test']

    show_one_sample(data_train[0])