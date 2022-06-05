import os


if __name__ == "__main__":
    path = '../Data/img/char_to_idx.txt'
    with open(path, 'r', encoding = 'utf-8') as file:
        char_to_idx = file.read()

    path = '../Data/img/oracle_source_img'
    sub_path = 'bnu_xxt_hard'
    cur_sub_path = os.path.join(path, sub_path)
    sub_name = 0
    for sample in os.listdir(cur_sub_path):
        os.rename(os.path.join(cur_sub_path, sample), os.path.join(cur_sub_path, '%d.jpg' % sub_name))
        sub_name += 1

    for sub_path in ('gbk_bronze_lst_seal', 'oracle_54081', 'other_font'):
        cur_path = os.path.join(path, sub_path)
        name = 0
        for character in os.listdir(cur_path):
            cur_sub_path = os.path.join(cur_path, character)
            sub_name = 0
            for sample in os.listdir(cur_sub_path):
                os.rename(os.path.join(cur_sub_path, sample), os.path.join(cur_sub_path, '%d.jpg' % sub_name))
                sub_name += 1
            os.rename(cur_sub_path, os.path.join(cur_path, '%d' % name))
            name += 1

    for shot in (1 ,3, 5):
        for mode in ('train', 'test'):
            path = '../Data/img/oracle_200_%d_shot/%s' % (shot, mode)
            for character in os.listdir(path):
                cur_sub_path = os.path.join(path, character)
                sub_name = 0
                for sample in os.listdir(cur_sub_path):
                    os.rename(os.path.join(cur_sub_path, sample), os.path.join(cur_sub_path, '%d.jpg' % sub_name))
                    sub_name += 1
                os.rename(cur_sub_path, os.path.join(path, '%d' % char_to_idx.index(character)))