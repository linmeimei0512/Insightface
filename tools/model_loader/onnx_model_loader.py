import sys
import onnx
import onnxruntime as onnxrt
import time

sys.path.append('../../')
from utils.exception_printer import exception_printer

'''
########################################
ONNX Model Loader
########################################
'''
class ONNXModelLoader:
    def __init__(self):
        print('\n*********** ONNX Model Loader ***********')
        print('onnx version: ', onnx.__version__)
        print('onnxruntime version: ', onnxrt.__version__)
        print('*****************************************')


    # ================================================
    # Load ONNX model
    #
    # :param onnx_model_path
    #
    # return onnx_model, onnx_sess
    # ================================================
    def load_onnx_model(self, onnx_model_path):
        print('\nStarting load ONNX model \'' + str(onnx_model_path) + '\'...')
        start_time = time.time()

        try:
            self.onnx_model = onnx.load(onnx_model_path)
            self.onnx_sess = onnxrt.InferenceSession(onnx_model_path)

            print('Input name: ', self.onnx_sess.get_inputs()[0].name)
            print('Output name: ', self.onnx_sess.get_outputs()[0].name)

            print('Load ONNX model success. Cost time: ' + str(time.time() - start_time) + 's.')

            return self.onnx_model, self.onnx_sess

        except Exception as ex:
            exception_printer('Load ONNX model failed.')
            return None