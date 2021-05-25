import sys
import os
import argparse
import cv2
import numpy as np
import torch
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from tools.model_loader.pytorch_model_loader import PyTorchModelLoader

try:
    from utils.image_loader import ImageLoader
    from utils.compare_util import Compare_Util, CompareDistanceType
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    from Insightface.utils.image_loader import ImageLoader
    from Insightface.utils.compare_util import Compare_Util, CompareDistanceType

np.set_printoptions(linewidth=2000)


# ================================================
# Load insightface pytorch model
#
# :param model_name
# :param model_path
# :param weight_path
#
# return insightface pytorch model
# ================================================
def load_insightface_pytorch_model(network, model_path, weight_path):
    pytorch_model_loader = PyTorchModelLoader()

    if model_path is None and network is not None:
        insightface_pytorch_model = pytorch_model_loader.load_insightface_pytorch_model(model_name=network,
                                                                                        pytorch_weight_path=weight_path)
        return insightface_pytorch_model

    else:
        return None


# ================================================
# Load image
#
# :param image_path
#
# return image
# ================================================
def load_image(image_path):
    image = ImageLoader.loader(image_path=image_path,
                               convert_color=cv2.COLOR_BGR2RGB,
                               transpose=(2, 0, 1),
                               dtype=np.float32)
    image = torch.Tensor(image).cuda()
    image.div_(255).sub_(0.5).div_(0.5)

    return image


# ================================================
# Get feature with face
#
# :param insightface_pytorch_model
# :param image_path
#
# return feature
# ================================================
def get_feature(insightface_pytorch_model:PyTorchModelLoader, image_path):
    image = load_image(image_path)
    feature = insightface_pytorch_model(image)
    feature = feature.cpu().detach().numpy()

    return feature[0]



'''
=============================
Main
=============================
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face recognition')
    parser.add_argument('--network', type=str, default='r100', help='backbone network.')
    parser.add_argument('--model_path', type=str, default=None, help='where is the insightface torch model.')
    parser.add_argument('--weight_path', type=str, default='../../model_zoo/Arcface/PyTorch/glint360k_cosface_r100_fp16_0.1/weight_backbone.pth', help='where is the insightface torch weight path.')
    parser.add_argument('--image_1_path', type=str, default='../../images/Tom_Hanks_01.jpg', help='where is the image 1 path.')
    parser.add_argument('--image_2_path', type=str, default='../../images/Tom_Hanks_02.jpg', help='where is the image 2 path.')
    args = parser.parse_args()

    print('********** Face Detection - Insightface **********')
    print('network: ' + str(args.network))
    print('model path: ' + str(args.model_path))
    print('weight path: ' + str(args.weight_path))
    print('image 1 path: ' + str(args.image_1_path))
    print('image 2 path: ' + str(args.image_2_path))
    print('**************************************************')

    # load insightface pytorch model
    insightface_pytorch_model = load_insightface_pytorch_model(args.network, args.model_path, args.weight_path)

    # check image is exist
    if not os.path.exists(args.image_1_path):
        print(str(args.image_1_path) + ' is not exist.')
        sys.exit()
    if not os.path.exists(args.image_2_path):
        print(str(args.image_2_path) + ' is not exist.')
        sys.exit()

    # get face feature
    start_time = time.time()
    feature_1 = get_feature(insightface_pytorch_model, args.image_1_path)
    feature_2 = get_feature(insightface_pytorch_model, args.image_2_path)

    # compare two features
    compare_utils = Compare_Util(False)
    _, _, cosine_distance, _, _ = compare_utils.compare_feature(CompareDistanceType.Cosine,
                                                                feature_1,
                                                                feature_2,
                                                                0.6, 0.6, 0.6)

    print('\nDistance: ' + str(cosine_distance) + '. Cost time: ' + str(time.time() - start_time))
