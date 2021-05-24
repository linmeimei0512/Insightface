import numpy as np
import cv2
import sys

sys.path.append('../../')
try:
    from utils.exception_printer import exception_printer
except ImportError:
    sys.path.append('../../../')
    from Insightface.utils.exception_printer import exception_printer

'''
########################################
Image Loader
########################################
'''
class ImageLoader:
    def __init__(self):
        return

    # ================================================
    # Loader
    # :param image_path
    # :param transpose
    # :param dtype
    #
    # return image
    # ================================================
    @staticmethod
    def loader(image_path, convert_color=None, transpose=(0, 1, 2), dtype=np.float32):
        try:
            # image = cv2.imread(image_path)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image = np.transpose(image, transpose)
            # image = np.array(image, dtype=dtype)
            # image = image / 255
            # image = np.array([image])

            image = cv2.imread(image_path)

            if convert_color is not None:
                image = cv2.cvtColor(image, convert_color)
            image = np.transpose(image, transpose)
            image = np.array(image, dtype=dtype)
            image = np.array([image])
            # image.div_(255).sub_(0.5).div_(0.5)

            return image

        except Exception as ex:
            exception_printer('Load image \'' + str(image_path) + '\' failed.')