import tensorflow as tf
import time
import sys

sys.path.append('../../')
from utils.exception_printer import exception_printer

'''
########################################
Tensorflow PB Model Loader
########################################
'''
class TensorflowPBModelLoader:
    def __init__(self):
        print('\n*********** Tensorflow PB Model Loader ***********')
        print('tensorflow version: ', tf.__version__)
        print('*************************************************')


    # ================================================
    # Load Tensorflow pb model
    #
    # :param tensorflow_pb_model_path
    #
    # return tensorflow_pb_model_sess
    #        tensorflow_input
    #        tensorflow_output
    # ================================================
    def load_tensorflow_pb_model(self, tensorflow_pb_model_path, input_name, output_name):
        print('\nStarting load tensorflow pb model \'' + str(tensorflow_pb_model_path) + '\'...')
        start_time = time.time()

        try:
            with open(tensorflow_pb_model_path, 'rb') as f:
                output_graph_def = tf.compat.v1.GraphDef()
                output_graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(output_graph_def, name="")

            self.tensorflow_pb_model_sess = tf.compat.v1.Session()
            self.tensorflow_pb_model_sess.as_default()

            self.input = self.tensorflow_pb_model_sess.graph.get_tensor_by_name(input_name)
            self.output = self.tensorflow_pb_model_sess.graph.get_tensor_by_name(output_name)

            print('Load tensorflow pb model success. Cost time: ', time.time() - start_time)
            return self.tensorflow_pb_model_sess, self.input, self.output

        except Exception as ex:
            exception_printer('Load tensorflow pb model failed.')
            return None