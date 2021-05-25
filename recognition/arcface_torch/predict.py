from __future__ import absolute_import
import sys
import os
import torch
import numpy as np
import cv2

sys.path.append('../../')
from tools.model_loader.pytorch_model_loader import PyTorchModelLoader
sys.path.append('../../../')
from Insightface.utils.image_loader import ImageLoader
from Insightface.utils.compare_util import Compare_Util, CompareDistanceType

np.set_printoptions(linewidth=2000)

pytorch_model_path = '../../../../Deep_Learning/InsightFace/Python/Models/PyTorch/emore_mask_cosface_r18_fp16/13_backbone.pth'
# pytorch_model_path = '../../../../Deep_Learning/InsightFace/Python/Models/PyTorch/glint360k_cosface_r18/0_backbone.pth'

# pytorch_weight_path = '../../../../Deep_Learning/InsightFace/Python/Models/PyTorch/glint360k_cosface_r50_fp16_0_1/backbone.pth'
# pytorch_weight_path = '../../../../Deep_Learning/InsightFace/Python/Models/PyTorch/emore_mask_cosface_r50_fp16_0_1/1_weight_backbone.pth'

pytorch_weight_path = '../../../../Deep_Learning/InsightFace/Python/Models/PyTorch/glint360k_cosface_r100_fp16_0_1/backbone.pth'
# pytorch_weight_path = '../../../../Deep_Learning/InsightFace/Python/Models/PyTorch/emore_mask_cosface_r100_fp16_0_1/0_weight_backbone.pth'
# pytorch_weight_path = None
input_shape = (3, 112, 112)

image1_path = '../../images/alan07.jpg'
image2_path = '../../images/alan06.jpg'
image3_path = '../../images/charlie01.jpg'

pytorch_model_loader = PyTorchModelLoader()
pytorch_model = pytorch_model_loader.load_insightface_pytorch_model(model_name='r100',
                                                                    pytorch_weight_path=pytorch_weight_path)
# pytorch_model = pytorch_model_loader.load_insightface_pytorch_model(model_name='r50',
#                                                                     pytorch_model_path=pytorch_model_path)

image1 = ImageLoader.loader(image_path=image1_path,
                            convert_color=cv2.COLOR_BGR2RGB,
                            transpose=(2, 0, 1),
                            dtype=np.float32)
image1_torch = torch.Tensor(image1).cuda()
image1_torch.div_(255).sub_(0.5).div_(0.5)

feature_by_pytorch_1 = pytorch_model(image1_torch)
feature_by_pytorch_1 = feature_by_pytorch_1.cpu().detach().numpy()
print('Feature by PyTorch 1 ' + str(feature_by_pytorch_1[0].shape) + ': \n', feature_by_pytorch_1[0])


image2 = ImageLoader.loader(image_path=image2_path,
                            convert_color=cv2.COLOR_BGR2RGB,
                            transpose=(2, 0, 1),
                            dtype=np.float32)
image2_torch = torch.Tensor(image2).cuda()
image2_torch.div_(255).sub_(0.5).div_(0.5)

feature_by_pytorch_2 = pytorch_model(image2_torch)
feature_by_pytorch_2 = feature_by_pytorch_2.cpu().detach().numpy()
print('Feature by PyTorch 2 ' + str(feature_by_pytorch_2[0].shape) + ': \n', feature_by_pytorch_2[0])

compare_utils = Compare_Util(False)
_, _, cosine_distance, _, _ = compare_utils.compare_feature(CompareDistanceType.Cosine,
                                                            feature_by_pytorch_1[0],
                                                            feature_by_pytorch_2[0],
                                                            0.6, 0.6, 0.6)
print('Distance: ', cosine_distance)


image3 = ImageLoader.loader(image_path=image3_path,
                            convert_color=cv2.COLOR_BGR2RGB,
                            transpose=(2, 0, 1),
                            dtype=np.float32)
image3_torch = torch.Tensor(image3).cuda()
image3_torch.div_(255).sub_(0.5).div_(0.5)

feature_by_pytorch_3 = pytorch_model(image3_torch)
feature_by_pytorch_3 = feature_by_pytorch_3.cpu().detach().numpy()
print('\nFeature by PyTorch 3 ' + str(feature_by_pytorch_3[0].shape) + ': \n', feature_by_pytorch_3[0])

compare_utils = Compare_Util(False)
_, _, cosine_distance, _, _ = compare_utils.compare_feature(CompareDistanceType.Cosine,
                                                            feature_by_pytorch_1[0],
                                                            feature_by_pytorch_3[0],
                                                            0.6, 0.6, 0.6)
print('Distance: ', cosine_distance)
