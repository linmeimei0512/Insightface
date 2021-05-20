import sys
import torch
import time
import numpy as np
import argparse

sys.path.append('../../')
from tools.model_loader.pytorch_model_loader import PyTorchModelLoader
from utils.image_loader import ImageLoader
from utils.exception_printer import exception_printer
from utils.compare_util import Compare_Util, CompareDistanceType

'''
########################################
PyTorch To ONNX Converter
########################################
'''
class PyTorchToONNXConverter:
    def __init__(self):
        self.init_device()

        print('\n*********** PyTorch To ONNX Converter ***********')
        print('torch version: ', torch.__version__)
        print('torch cuda is available: ', torch.cuda.is_available())
        print('torch device: ', self.device)
        print('*************************************************')


    # ================================================
    # Initialize device
    # ================================================
    def init_device(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # ================================================
    # Convert to ONNX model
    #
    # :param pytorch_model_path
    # :param pytorch_weight_path
    # :param input_shape
    # :param onnx_model_output_path
    # :param input_names
    # :param output_names
    # ================================================
    def convert(self, pytorch_model_path, pytorch_weight_path, input_shape, onnx_model_output_path, input_names, output_names):
        self.load_pytorch_model(pytorch_model_path, pytorch_weight_path, input_shape)

        start_time = time.time()
        print('\nStarting convert to ONNX model ...')
        try:
            image = torch.empty(size=(1, *input_shape), dtype=torch.float, device=self.device)

            torch.onnx.export(model=self.pytorch_model,
                              args=image,
                              f=onnx_model_output_path,
                              verbose=False,
                              input_names=input_names,
                              output_names=output_names)

            print('ONNX model export success, saved as ' + str(onnx_model_output_path) + '. Cost time: ' + str(time.time() - start_time) + 's.')

        except Exception as ex:
            exception_printer('Convert to ONNX model failed.')


    # ================================================
    # Load pytorch model
    #
    # :param pytorch_model_path
    # :param pytorch_weight_path
    # :param input_shape
    # ================================================
    def load_pytorch_model(self, pytorch_model_path, pytorch_weight_path, input_shape):
        self.pytorch_model_loader = PyTorchModelLoader()
        self.pytorch_model = self.pytorch_model_loader.load_pytorch_model(pytorch_model_path=pytorch_model_path,
                                                                          pytorch_weight_path=pytorch_weight_path,
                                                                          input_shape=input_shape,
                                                                          train=False)

    # ================================================
    # Test ONNX model by PyTorch model
    #
    # :param test_image_path
    # :param onnx_model_sess
    # :param pytorch_model
    # ================================================
    def test_onnx_model_by_pytorch_model(self, test_image_path, onnx_model_sess, pytorch_model):
        print('\nStarting test ONNX model by PyTorch model ...')
        start_time = time.time()

        try:
            image = ImageLoader.loader(image_path=test_image_path,
                                       transpose=(2, 0, 1),
                                       dtype=np.float32)
            print('Test image \'' + str(test_image_path) + '\', shape: ' + str(image.shape))

            # ONNX
            input_name = onnx_model_sess.get_inputs()[0].name
            feature_by_onnx = onnx_model_sess.run(None, {input_name: image})
            feature_by_onnx = np.array(feature_by_onnx[0])
            print('Feature by ONNX ' + str(feature_by_onnx[0].shape) + ': \n', feature_by_onnx[0])

            # PyTorch
            image_torch = torch.Tensor(image).cuda()
            feature_by_pytorch = pytorch_model(image_torch)
            feature_by_pytorch = feature_by_pytorch.cpu().detach().numpy()
            print('Feature by PyTorch ' + str(feature_by_pytorch[0].shape) + ': \n', feature_by_pytorch[0])

            compare_utils = Compare_Util(False)
            _, _, cosine_distance, _, _ = compare_utils.compare_feature(CompareDistanceType.Cosine,
                                                                        feature_by_onnx[0],
                                                                        feature_by_pytorch[0],
                                                                        0.6, 0.6, 0.6)
            print('Distance: ', cosine_distance)
            print('Test finish. Cost time: ' + str(time.time() - start_time) + 's.')

        except Exception as ex:
            exception_printer('Test ONNX model by PyTorch model failed.')



'''
=============================
Default
=============================
'''
pytorch_model_path = '../../../../Deep_Learning/InsightFace/Python/Models/PyTorch/emore_mask_resnet_r18/6_backbone.pth'
pytorch_weight_path = None
input_shape = '3,112,112'
onnx_model_output_path = '../../../../Deep_Learning/InsightFace/Python/Models/PyTorch/emore_mask_resnet_r18/emore_mask_r18.onnx'
input_names = 'input'
output_names = 'output'


'''
=============================
Main
=============================
'''
if __name__ == '__main__':
    np.set_printoptions(linewidth=2000, precision=20)

    parser = argparse.ArgumentParser(description='PyTorch model convert to ONNX model')
    parser.add_argument('--pytorch_model_path', default=pytorch_model_path, help='where is the pytorch model.')
    parser.add_argument('--pytorch_weight_path', default=pytorch_weight_path, help='where is the pytorch weight.')
    parser.add_argument('--input_shape', default=input_shape, help='input shape for pytorch model. ex. 3,112,112')
    parser.add_argument('--onnx_model_output_path', default=onnx_model_output_path, help='where is save to output onnx model.')
    parser.add_argument('--input_names', default=input_names, help='the input names to use for onnx model. ex. input')
    parser.add_argument('--output_names', default=output_names, help='the output name to use for onnx model. ex. output')
    args = parser.parse_args()

    pytorch_model_path = args.pytorch_model_path
    pytorch_weight_path = args.pytorch_weight_path
    input_shape = [int(x) for x in args.input_shape.split(',')]
    input_shape = (input_shape[0], input_shape[1], input_shape[2])
    onnx_model_output_path = args.onnx_model_output_path
    input_names = [str(x) for x in args.input_names.split(',')]
    output_names = [str(x) for x in args.output_names.split(',')]

    print('\n************** PyTorch model convert to ONNX model **************')
    print('pytorch model path: ' + str(pytorch_model_path))
    print('pytorch weight path: ' + str(pytorch_weight_path))
    print('input shape: ' + str(input_shape))
    print('onnx model output path: ' + str(onnx_model_output_path))
    print('input names: ' + str(input_names))
    print('output names: ' + str(output_names))

    pytorch_to_onnx_converter = PyTorchToONNXConverter()
    pytorch_to_onnx_converter.convert(pytorch_model_path=pytorch_model_path,
                                      pytorch_weight_path=pytorch_weight_path,
                                      input_shape=input_shape,
                                      onnx_model_output_path=onnx_model_output_path,
                                      input_names=input_names,
                                      output_names=output_names)