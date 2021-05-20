import sys
from onnx_tf.backend import prepare
import tensorflow as tf
import pkg_resources
import time
import numpy as np
import argparse

sys.path.append('../../')
from tools.model_loader.onnx_model_loader import ONNXModelLoader
from utils.image_loader import ImageLoader
from utils.exception_printer import exception_printer
from utils.compare_util import Compare_Util, CompareDistanceType

'''
########################################
ONNX To Tensorflow PB Converter
########################################
'''
class ONNXToTensorflowPBConverter:
    def __init__(self):
        print('\n*********** ONNX To Tensorflow PB Converter ***********')
        print('onnx_tf version: ', pkg_resources.get_distribution('onnx_tf').version)
        print('tensorflow version: ', tf.__version__)
        print('*******************************************************')


    # ================================================
    # Convert to tensorflow pb model
    #
    # :param onnx_model_path
    # :param tensorflow_pb_model_output_path
    # ================================================
    def convert(self, onnx_model_path, tensorflow_pb_model_output_path):
        self.load_onnx_model(onnx_model_path)

        start_time = time.time()
        print('\nStarting convert to tensorflow pb model ...')
        try:
            onnx_tf_exporter = prepare(self.onnx_model)
            onnx_tf_exporter.export_graph(tensorflow_pb_model_output_path)

            print('Tensorflow export success, saved as ' + str(tensorflow_pb_model_output_path) + '. Cost time: ' + str(time.time() - start_time))

        except Exception as ex:
            exception_printer('Convert to tensorflow pb model failed.')


    # ================================================
    # Load ONNX model
    #
    # :param onnx_model_path
    # ================================================
    def load_onnx_model(self, onnx_model_path):
        self.onnx_model_loader = ONNXModelLoader()
        self.onnx_model, self.onnx_sess = self.onnx_model_loader.load_onnx_model(onnx_model_path=onnx_model_path)


    # ================================================
    # Test Tensorflow pb model by ONNX model
    #
    # :param test_image_path
    # :param tensorflow_pb_model
    # :param tensorflow_input
    # :param tensorflow_output
    # :param onnx_model_sess
    # ================================================
    def test_tensorflow_pb_model_by_onnx_model(self, test_image_path, tensorflow_pb_model, tensorflow_input, tensorflow_output, onnx_model_sess):
        print('\nStarting test Tensorflow pb model by ONNX model ...')
        start_time = time.time()

        try:
            image = ImageLoader.loader(image_path=test_image_path,
                                       transpose=(2, 0, 1),
                                       dtype=np.float32)
            print('Test image \'' + str(test_image_path) + '\', shape: ' + str(image.shape))

            # Tensorflow pb
            feature_by_tensorflow = tensorflow_pb_model.run(tensorflow_output, feed_dict={tensorflow_input: image})
            print('Feature by tensorflow ' + str(feature_by_tensorflow.shape) + '\n', feature_by_tensorflow)

            # ONNX
            input_name = onnx_model_sess.get_inputs()[0].name
            feature_by_onnx = onnx_model_sess.run(None, {input_name: image})
            feature_by_onnx = np.array(feature_by_onnx[0])
            print('Feature by ONNX ' + str(feature_by_onnx[0].shape) + ': \n', feature_by_onnx[0])

            compare_utils = Compare_Util(False)
            _, _, cosine_distance, _, _ = compare_utils.compare_feature(CompareDistanceType.Cosine,
                                                                        feature_by_tensorflow[0],
                                                                        feature_by_onnx[0],
                                                                        0.6, 0.6, 0.6)
            print('Distance: ', cosine_distance)
            print('Test finish. Cost time: ' + str(time.time() - start_time) + 's.')

        except Exception as ex:
            exception_printer('Test Tensorflow pb model by ONNX model failed.')



'''
=============================
Default
=============================
'''
onnx_model_path = '../../../../Deep_Learning/InsightFace/Python/Models/PyTorch/emore_mask_resnet_r18/emore_mask_r18.onnx'
tensorflow_pb_model_output_path = '../../../../Deep_Learning/InsightFace/Python/Models/PyTorch/emore_mask_resnet_r18/emore_mask_r18.pb'



'''
=============================
Main
=============================
'''
if __name__ == '__main__':
    np.set_printoptions(linewidth=2000, precision=20)

    parser = argparse.ArgumentParser(description='ONNX model convert to tensorflow pb model')
    parser.add_argument('--onnx_model_path', default=onnx_model_path, help='where is the onnx model.')
    parser.add_argument('--tensorflow_pb_model_output_path', default=tensorflow_pb_model_output_path, help='where is save to tensorflow pb model.')
    args = parser.parse_args()

    onnx_model_path = args.onnx_model_path
    tensorflow_pb_model_output_path = args.tensorflow_pb_model_output_path

    print('\n************** ONNX model convert to Tensorflow pb model **************')
    print('onnx model path: ' + str(onnx_model_path))
    print('tensorflow pb model output path: ' + str(tensorflow_pb_model_output_path))

    onnx_to_tensorflow_pb_converter = ONNXToTensorflowPBConverter()
    onnx_to_tensorflow_pb_converter.convert(onnx_model_path=onnx_model_path,
                                            tensorflow_pb_model_output_path=tensorflow_pb_model_output_path)
