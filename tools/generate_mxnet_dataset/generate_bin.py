import numpy as np
import argparse
import pickle
import os
import psutil
import time

'''
=============================
Default
=============================
'''
pairs_txt_path = '../../../../Deep_Learning/InsightFace/Python/Dataset/faces_emore_mask/pairs.txt'
output_bin_path = '../../../../Deep_Learning/InsightFace/Python/Dataset/faces_emore_mask/faces_emore_mask.bin'


'''
=============================
Bin Generator
=============================
'''
class BinGenerator:
    div_gb_factor = (1024.0 ** 3)

    """
    :param pairs_txt_path:      where is the pairs.txt.
    :param output_bin_path:     where to save the output .bin.
    """
    def __init__(self, pairs_txt_path, output_bin_path):
        self.pairs_txt_path = pairs_txt_path
        self.output_bin_path = output_bin_path


    # ===============================
    # Start generate
    # ===============================
    def generate(self):
        if os.path.isfile(self.output_bin_path):
            remove = input('\nThe %s is exist. Will you remove old .bin? (yes/no)' %self.output_bin_path)

            if remove.lower() == 'yes':
                os.remove(self.output_bin_path)
            else:
                return

        start_time = time.time()
        pairs = self.read_pairs(pairs_path=self.pairs_txt_path)
        paths, issame_list = self.get_paths(pairs=pairs)
        bins = []

        i = 0
        for path in paths:
            with open(path, 'rb') as fin:
                bin = fin.read()
                bins.append(bin)
                i += 1

            if i % 1000 == 0:
                print('loading dataset', i)

                pc_mem = psutil.virtual_memory()
                print("total memory: %f GB" % float(pc_mem.total / self.div_gb_factor))
                print("used memory: %f GB" % float(pc_mem.used / self.div_gb_factor))
                print("available memory: %f GB\n" % float(pc_mem.available / self.div_gb_factor))

        # write to .bin
        with open(self.output_bin_path, 'wb') as f:
            pickle.dump((bins, issame_list), f, protocol=pickle.HIGHEST_PROTOCOL)
            print('Generate over, save as ' + str(self.output_bin_path) + '. Time cost: ' + str(time.time() - start_time) + ' sec.')


    # ===============================
    # Initialize image size
    # ===============================
    def init_image_size(self, image_size_str=str):
        image_size = []
        for size in image_size_str.split(','):
            image_size.append(int(size))

        return image_size


    # ===============================
    # Read pairs.txt
    # ===============================
    def read_pairs(self, pairs_path):
        pairs = []
        with open(pairs_path, 'r') as f:
            for line in f.readlines()[1:]:
                pair = line.strip().split()
                pairs.append(pair)

        return np.array(pairs)


    # ===============================
    # Get paths
    # ===============================
    def get_paths(self, pairs):
        nrof_skipped_pairs = 0
        path_list = []
        issame_list = []

        for pair in pairs:
            path0 = pair[0]
            path1 = pair[1]

            if pair[2] == '1':
                issame = True
            else:
                issame = False

            if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
                path_list += (path0, path1)
                issame_list.append(issame)
            else:
                print('not exists', path0, path1)
                nrof_skipped_pairs += 1

        if nrof_skipped_pairs > 0:
            print('Skipped %d image pairs' % nrof_skipped_pairs)

        return path_list, issame_list


'''
=============================
Main
=============================
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate pin')
    parser.add_argument('--pairs_txt_path', default=pairs_txt_path, help='where is the pairs.txt. ')
    parser.add_argument('--output_bin_path', default=output_bin_path, help='where to save the output .bin. ')
    args = parser.parse_args()

    pairs_txt_path = args.pairs_txt_path
    output_bin_path = args.output_bin_path

    print('************** Generate .bin **************')
    print('pairs.txt path: ' + str(pairs_txt_path))
    print('output path: ' + str(output_bin_path))
    print('\n')

    bin_generator = BinGenerator(pairs_txt_path=pairs_txt_path,
                                 output_bin_path=output_bin_path)
    bin_generator.generate()