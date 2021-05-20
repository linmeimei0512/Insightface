import numpy as np
import sys
import os
import argparse

sys.path.append('../../')
from tools.model_converter.pytorch_to_onnx_converter import PyTorchToONNXConverter
from tools.model_converter.onnx_to_pb_converter import ONNXToTensorflowPBConverter
from tools.model_converter.pb_to_tflite_converter import TensorflowPBToTensorflowLiteConverter


'''
=============================
Default
=============================
'''
output_model_root_path = ''
output_model_name = ''

pytorch_model_path = '../../../../Deep_Learning/InsightFace/Python/Models/PyTorch/emore_mask_resnet_r18/6_backbone.pth'
pytorch_weight_path = None

input_shape = '3,112,112'
input_names = 'input'
output_names = 'output'

onnx_model_output_path = ''
pb_model_output_path = ''
tensorflow_lite_model_output_path = '../../../../Deep_Learning/InsightFace/Python/Models/PyTorch/emore_mask_resnet_r18/emore_mask_r18.tflite'


'''
Get output model root path
'''
def get_output_model_root_path():
    global output_model_root_path
    model_root_path_list = tensorflow_lite_model_output_path.split('/')
    for i in range(len(model_root_path_list) - 1):
        output_model_root_path = os.path.join(output_model_root_path, model_root_path_list[i])


'''
Get output model name
'''
def get_output_model_name():
    global output_model_name
    model_root_path_list = tensorflow_lite_model_output_path.split('/')
    output_model_name = model_root_path_list[len(model_root_path_list)-1]
    output_model_name = output_model_name.split('.')
    output_model_name = output_model_name[0]


'''
PyTorch to ONNX
'''
def pytorch_to_onnx():
    global onnx_model_output_path
    onnx_model_output_path = os.path.join(output_model_root_path, output_model_name + '.onnx')

    pytorch_to_onnx_converter = PyTorchToONNXConverter()
    pytorch_to_onnx_converter.convert(pytorch_model_path=pytorch_model_path,
                                      pytorch_weight_path=pytorch_weight_path,
                                      input_shape=input_shape,
                                      onnx_model_output_path=onnx_model_output_path,
                                      input_names=input_names,
                                      output_names=output_names)

'''
ONNX to pb
'''
def onnx_to_pb():
    global pb_model_output_path
    pb_model_output_path = os.path.join(output_model_root_path, output_model_name + '.pb')

    onnx_to_tensorflow_pb_converter = ONNXToTensorflowPBConverter()
    onnx_to_tensorflow_pb_converter.convert(onnx_model_path=onnx_model_output_path,
                                            tensorflow_pb_model_output_path=pb_model_output_path)

'''
pb to tflite
'''
def pb_to_tflite():
    tensorflow_pb_to_tensorflow_lite_converter = TensorflowPBToTensorflowLiteConverter()
    tensorflow_pb_to_tensorflow_lite_converter.convert(tensorflow_pb_model_path=pb_model_output_path,
                                                       tensorflow_lite_model_output_path=tensorflow_lite_model_output_path,
                                                       input_names=input_names,
                                                       output_names=output_names,
                                                       quant=True)


'''
=============================
Main
=============================
'''
if __name__ == '__main__':
    np.set_printoptions(linewidth=2000, precision=20)

    parser = argparse.ArgumentParser(description='PyTorch model convert to tflite model')
    parser.add_argument('--pytorch_model_path', default=pytorch_model_path, help='where is the pytorch model.')
    parser.add_argument('--pytorch_weight_path', default=pytorch_weight_path, help='where is the pytorch weight.')
    parser.add_argument('--input_shape', default=input_shape, help='input shape for pytorch model. ex. 3,112,112')
    parser.add_argument('--input_names', default=input_names, help='the input names to use for onnx model. ex. input')
    parser.add_argument('--output_names', default=output_names, help='the output name to use for onnx model. ex. output')
    parser.add_argument('--tensorflow_lite_model_output_path', default=tensorflow_lite_model_output_path, help='where is save to tflite model.')
    args = parser.parse_args()

    pytorch_model_path = args.pytorch_model_path
    pytorch_weight_path = args.pytorch_weight_path
    input_shape = [int(x) for x in args.input_shape.split(',')]
    input_shape = (input_shape[0], input_shape[1], input_shape[2])
    input_names = [str(x) for x in args.input_names.split(',')]
    output_names = [str(x) for x in args.output_names.split(',')]
    tensorflow_lite_model_output_path = args.tensorflow_lite_model_output_path

    print('\n************** PyTorch model convert to tflite model **************')
    print('pytorch model path: ' + str(pytorch_model_path))
    print('pytorch weight path: ' + str(pytorch_weight_path))
    print('input shape: ' + str(input_shape))
    print('input names: ' + str(input_names))
    print('output names: ' + str(output_names))
    print('tensorflow lite model output path: ' + str(tensorflow_lite_model_output_path))

    get_output_model_root_path()
    get_output_model_name()
    pytorch_to_onnx()
    onnx_to_pb()
    pb_to_tflite()