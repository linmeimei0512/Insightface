import tensorflow as tf
import time
import numpy as np
import sys
import argparse

sys.path.append('../../')
from utils.image_loader import ImageLoader
from utils.exception_printer import exception_printer
from utils.compare_util import Compare_Util, CompareDistanceType

'''
########################################
Tensorflow PB To Tensorflow Lite Converter
########################################
'''
class TensorflowPBToTensorflowLiteConverter:
    def __init__(self):
        print('\n*********** Tensorflow PB To Tensorflow Lite Converter ***********')
        print('tensorflow version: ', tf.__version__)
        print('******************************************************************')


    # ================================================
    # Convert to tensorflow lte model
    #
    # :param tensorflow_pb_model_path
    # :param tensorflow_lite_model_output_path
    # :param input_names
    # :param output_names
    # :param quant
    # ================================================
    def convert(self, tensorflow_pb_model_path, tensorflow_lite_model_output_path, input_names, output_names, quant):
        print('\nStarting convert to tensorflow lite ...')
        start_time = time.time()

        try:
            converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(graph_def_file=tensorflow_pb_model_path,
                                                                            input_arrays=input_names,
                                                                            output_arrays=output_names)
            if quant:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                # converter.target_spec.supported_types = [tf.float16]
                # converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
                # converter.inference_type = tf.uint8
                converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
            tflite_model = converter.convert()

            with open(tensorflow_lite_model_output_path, 'wb') as f:
                f.write(tflite_model)

            print('Tensorflow lite export success, saved as ' + str(tensorflow_lite_model_output_path) + '. Cost time: ' + str(time.time() - start_time))

        except Exception as ex:
            exception_printer('Convert to tensorflow failed.')


    # ================================================
    # Test Tensorflow lite model by tensorflow pb model
    #
    # :param test_image_path
    # :param tensorflow_lite_model
    # :param tensorflow_lite_input
    # :param tensorflow_lite_output
    # :param tensorflow_pb_model
    # :param tensorflow_input
    # :param tensorflow_output
    # ================================================
    def test_tensorflow_lite_model_by_tensorflow_pb_model(self, test_image_path, tensorflow_lite_model, tensorflow_lite_input, tensorflow_lite_output, tensorflow_pb_model, tensorflow_input, tensorflow_output):
        print('\nStarting test tensorflow lite model by tensorflow pb model ...')
        start_time = time.time()

        try:
            image = ImageLoader.loader(image_path=test_image_path,
                                       transpose=(2, 0, 1),
                                       dtype=np.float32)
            print('Test image \'' + str(test_image_path) + '\', shape: ' + str(image.shape))

            # Tensorflow lite
            tensorflow_lite_model.set_tensor(tensorflow_lite_input[0]['index'], image)
            tensorflow_lite_model.invoke()
            feature_by_tensorflow_lite = tensorflow_lite_model.get_tensor(tensorflow_lite_output[0]['index'])
            print('Feature by tensorflow lite ' + str(feature_by_tensorflow_lite[0].shape) + ': \n', feature_by_tensorflow_lite[0])

            # Tensorflow pb
            feature_by_tensorflow = tensorflow_pb_model.run(tensorflow_output, feed_dict={tensorflow_input: image})
            print('Feature by tensorflow ' + str(feature_by_tensorflow.shape) + '\n', feature_by_tensorflow)

            compare_utils = Compare_Util(False)
            _, _, cosine_distance, _, _ = compare_utils.compare_feature(CompareDistanceType.Cosine,
                                                                        feature_by_tensorflow_lite[0],
                                                                        feature_by_tensorflow[0],
                                                                        0.6, 0.6, 0.6)
            print('Distance: ', cosine_distance)
            print('Test finish. Cost time: ' + str(time.time() - start_time) + 's.')

        except Exception as ex:
            exception_printer('Test tensorflow lite model by tensorflow pb model failed.')



'''
=============================
Default
=============================
'''
tensorflow_pb_model_path = '../../../../Deep_Learning/InsightFace/Python/Models/PyTorch/emore_mask_resnet_r18/emore_mask_r18.pb'
tensorflow_lite_model_output_path = '../../../../Deep_Learning/InsightFace/Python/Models/PyTorch/emore_mask_resnet_r18/emore_mask_r18.tflite'
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
    parser.add_argument('--tensorflow_pb_model_path', default=tensorflow_pb_model_path, help='where is the tensorflow pb model.')
    parser.add_argument('--tensorflow_lite_model_output_path', default=tensorflow_lite_model_output_path, help='where is save to tflite model.')
    parser.add_argument('--input_names', default=input_names, help='the input names for pb model. ex. input')
    parser.add_argument('--output_names', default=output_names, help='the output name for pb model. ex. output')
    args = parser.parse_args()

    tensorflow_pb_model_path = args.tensorflow_pb_model_path
    tensorflow_lite_model_output_path = args.tensorflow_lite_model_output_path
    input_names = [str(x) for x in args.input_names.split(',')]
    output_names = [str(x) for x in args.output_names.split(',')]

    print('\n************** Tensorflow pb model convert to Tensorflow Lite (tflite) model **************')
    print('tensorflow pb model path: ' + str(tensorflow_pb_model_path))
    print('tensorflow lite model output path: ' + str(tensorflow_lite_model_output_path))
    print('input names: ' + str(input_names))
    print('output names: ' + str(output_names))

    tensorflow_pb_to_tensorflow_lite_converter = TensorflowPBToTensorflowLiteConverter()
    tensorflow_pb_to_tensorflow_lite_converter.convert(tensorflow_pb_model_path=tensorflow_pb_model_path,
                                                       tensorflow_lite_model_output_path=tensorflow_lite_model_output_path,
                                                       input_names=input_names,
                                                       output_names=output_names,
                                                       quant=True)