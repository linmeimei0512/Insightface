import os
import sys
import datetime
import numpy as np
import sklearn
import argparse
from tqdm import tqdm
from typing import List

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from recognition import verification
from tools.model_loader.onnx_model_loader import ONNXModelLoader
from tools.bin_loader.bin_loader import BinLoader
from utils.loading_animation import LoadingAnimation

'''
########################################
Verification ONNX
########################################
'''
class VerificationONNX:
    def __init__(self, onnx_model_path, bin_path_list, frequent, rank, image_size=(112, 112)):
        self.frequent: int = frequent
        self.rank: int = rank
        self.highest_acc: float = 0.0
        self.ver_list: List[object] = []
        self.ver_name_list: List[str] = []

        self.init_onnx_model(onnx_model_path=onnx_model_path)
        self.init_bin_dataset(bin_path_list=bin_path_list, image_size=image_size)

    def init_onnx_model(self, onnx_model_path):
        '''
        Initialize ONNX model

        Args:
            onnx_model_path:
        '''
        onnx_model_loader = ONNXModelLoader()
        self.onnx_model, self.onnx_sess = onnx_model_loader.load_onnx_model(onnx_model_path=onnx_model_path)


    def init_bin_dataset(self, bin_path_list, image_size):
        '''
        Initialize bin dataset

        Args:
            bin_path_list:
            image_size:
        '''
        bin_loader = BinLoader()

        for bin_path in bin_path_list:
            data_set = bin_loader.load(bin_path=bin_path, image_size=image_size)
            self.ver_list.append(data_set)


    def verify(self):
        '''
        Verify model
        '''
        data_list = self.ver_list[0][0]
        issame_list = self.ver_list[0][1]
        embeddings_list = []
        time_consumed = 0.0
        batch_size = 1
        nfolds = 10

        print('\n')
        loading_animate = LoadingAnimation(title='Verify ...')
        loading_animate.start()
        for i in range(len(data_list)):
            data = data_list[i]
            embeddings = None
            ba = 0
            while ba < data.shape[0]:
                bb = min(ba + batch_size, data.shape[0])
                count = bb - ba
                _data = data[bb - batch_size: bb]
                time0 = datetime.datetime.now()
                img = ((_data / 255) - 0.5) / 0.5
                _embeddings = self.onnx_sess.run(None, {'input': img.numpy()})
                _embeddings = np.array(_embeddings[0])
                time_now = datetime.datetime.now()
                diff = time_now - time0
                time_consumed += diff.total_seconds()
                if embeddings is None:
                    embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
                embeddings[ba:bb, :] = _embeddings[(batch_size - count):, :]
                ba = bb

            embeddings_list.append(embeddings)

        _xnorm = 0.0
        _xnorm_cnt = 0
        for embed in embeddings_list:
            for i in range(embed.shape[0]):
                _em = embed[i]
                _norm = np.linalg.norm(_em)
                _xnorm += _norm
                _xnorm_cnt += 1
        _xnorm /= _xnorm_cnt

        embeddings = embeddings_list[0].copy()
        embeddings = sklearn.preprocessing.normalize(embeddings)
        acc1 = 0.0
        std1 = 0.0
        embeddings = embeddings_list[0] + embeddings_list[1]
        embeddings = sklearn.preprocessing.normalize(embeddings)
        # print(embeddings.shape)
        loading_animate.end()
        print('\ninfer time', time_consumed)
        _, _, accuracy, val, val_std, far = verification.evaluate(embeddings, issame_list, nrof_folds=nfolds)
        acc2, std2 = np.mean(accuracy), np.std(accuracy)

        print('Accuracy-Flip: %1.5f+-%1.5f' % (acc2, std2))
        # print(acc1)
        # print(std1)
        # print(acc2)
        # print(std2)
        # print(_xnorm)




'''
=============================
Default
=============================
'''
# onnx_model_path = '../../model_zoo/Arcface/PyTorch/glint360k_cosface_r18_fp16_0.1/glint360k_cosface_r18_fp16_0.onnx'          # Accuracy-Flip: 0.94033+-0.00878   0.96517+-0.00664    0.93633+-0.00951    0.97200+-0.00627
# onnx_model_path = '../../model_zoo/Arcface/PyTorch/glint360k_cosface_r100_fp16_0.1/insightface-glint360k-r100.onnx'           # Accuracy-Flip: 0.98633+-0.00194   0.98917+-0.00496    0.97467+-0.00600    0.98650+-0.00431
onnx_model_path = '../../model_zoo/Arcface/PyTorch/emore_mask_cosface_r100_fp16_0.1/17013-1199581-00-insightface-r100.onnx'   # Accuracy-Flip: 0.97317+-0.00555   0.97817+-0.00474    0.95600+-0.01033    0.97583+-0.00271
# onnx_model_path = '../../model_zoo/Arcface/PyTorch/emore_mask_cosface_r100_fp16_0.1/72220-4916348-00-insightface-r100.onnx'   # Accuracy-Flip: 0.95433+-0.00564   0.97550+-0.00415    0.94650+-0.01107    0.97550+-0.00415
# onnx_model_path = '../../model_zoo/Arcface/PyTorch/emore_mask_cosface_r100_fp16_0.1/17013-1199581-01-insightface-r100.onnx'   # Accuracy-Flip: 0.95433+-0.00564                       0.95600+-0.01033
# bin_path = '../../../Datasets/faces_emore/faces_emore_mask.bin'
bin_path = '../../../Datasets/test/msi_female.bin'

'''
=============================
Main
=============================
'''
if __name__ == '__main__':
    np.set_printoptions(linewidth=2000, threshold=10, edgeitems=10)

    parser = argparse.ArgumentParser(description='Verify ONNX')
    parser.add_argument('-m', '--onnx_model_path', type=str, default=onnx_model_path, help='where is the onnx model.')
    parser.add_argument('-b', '--bin_path', type=str, default=bin_path, help='where is the bin file.')
    args = parser.parse_args()

    onnx_model_path = args.onnx_model_path
    bin_path = args.bin_path

    print('\n********** Verify ONNX **********')
    print('ONNX model path: ', onnx_model_path)
    print('bin path: ', bin_path)
    print('*********************************\n')

    verification_onnx = VerificationONNX(onnx_model_path=onnx_model_path,
                                         bin_path_list=[bin_path],
                                         frequent=2000,
                                         rank=0)
    verification_onnx.verify()