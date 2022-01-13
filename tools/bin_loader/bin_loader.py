import os
import pickle
import mxnet as mx
import torch
import tqdm
import time
from mxnet import ndarray as nd

'''
########################################
Bin Loader
########################################
'''
class BinLoader:
    def load(self, bin_path, image_size):
        '''
        Load bin file

        Args:
            bin_path:
            image_size:

        Returns:
            date_list
            issame_list
        '''
        print('\n********** Bin Loader **********')
        print('Bin file path: ' + bin_path)
        print('Image size: ' + str(image_size))

        # Check bin file is exist
        if not os.path.exists(bin_path):
            print('\nError bin file (' + bin_path + ') is not exist!')

        # Loading bin file
        print('\nOpen bin file ...')
        try:
            # Python 2
            with open(bin_path, 'rb') as f:
                bins, issame_list = pickle.load(f)
        except UnicodeDecodeError as e:
            # Python 3
            with open(bin_path, 'rb') as f:
                bins, issame_list = pickle.load(f, encoding='bytes')

        # Get same and different number
        same_num = 0
        diff_num = 0
        for issame in issame_list:
            if issame:
                same_num += 1
            else:
                diff_num += 1
        print('Same number: ' + str(same_num))
        print('Different number: ' + str(diff_num))

        data_list = []
        for flip in [0, 1]:
            data = torch.empty((len(issame_list) * 2, 3, image_size[0], image_size[1]))
            data_list.append(data)

        time.sleep(1)
        progress = tqdm.tqdm(total=len(issame_list) * 2)
        progress.set_description('Loading ...')
        for idx in range(len(issame_list) * 2):
            img = mx.image.imdecode(bins[idx])
            # Resize image
            if img.shape[1] != image_size[0]:
                img = mx.image.resize_short(img, image_size[0])
            img = nd.transpose(img, axes=(2, 0, 1))
            for flip in [0, 1]:
                if flip == 1:
                    img = mx.ndarray.flip(data=img, axis=2)
                data_list[flip][idx][:] = torch.from_numpy(img.asnumpy())
            progress.update(1)

        progress.close()
        time.sleep(1)
        print('********** Bin Loader Finish *********')
        return data_list, issame_list



'''
=============================
Main
=============================
'''
if __name__ == '__main__':
    bin_loader = BinLoader()
    bin_loader.load(bin_path='../../../Datasets/test/agedb_30.bin', image_size=[112, 112])