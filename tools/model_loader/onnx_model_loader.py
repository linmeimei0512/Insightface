import os
import sys
import cv2
import numpy as np
import onnx
import onnxruntime as onnxrt
import time

try:
    from utils.image_loader import ImageLoader
    from utils.exception_printer import exception_printer
    from utils.loading_animation import LoadingAnimation
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    from Insightface.utils.image_loader import ImageLoader
    from Insightface.utils.exception_printer import exception_printer

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
            self.input_name = self.onnx_sess.get_inputs()[0].name
            self.output_name = self.onnx_sess.get_outputs()[0].name

            print('Input name: ', self.input_name)
            print('Output name: ', self.output_name)
            print('Load ONNX model success. Cost time: ' + str(time.time() - start_time) + 's.')

            return self.onnx_model, self.onnx_sess

        except Exception as ex:
            exception_printer('Load ONNX model failed.')
            return None

    def predict(self, image_path=None, image=None, type=None):
        '''
        Predict

        Args:
            image_path:
            type: image type

        Returns:
            feature by onnx model
        '''
        print('\n')
        loading_animation = LoadingAnimation(title='Predict ...')
        loading_animation.start()

        if image is None:
            image = self.__load_image(image_path=image_path, type=type)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.transpose(image, (2, 0, 1))
            image = np.array(image, dtype=np.float32)
            image = np.array([image])
            image = (image / 255 - 0.5) / 0.5

        if image is None:
            return None

        output = self.onnx_sess.run(None, {self.input_name: image})
        output = np.array(output[0])

        loading_animation.end()
        return output

    def __load_image(self, image_path, type):
        '''
        Load image for onnx use

        Args:
            image_path:
            type:

        Returns:
            image array for onnx use
        '''
        image = ImageLoader.loader(image_path=image_path,
                                   convert_color=type,
                                   transpose=(2, 0, 1),
                                   dtype=np.float32)
        if image is not None:
            image = (image / 255 - 0.5) / 0.5
        return image