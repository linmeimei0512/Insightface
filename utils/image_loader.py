import numpy as np
import cv2

from utils.exception_printer import exception_printer

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
    def loader(image_path, transpose=(0, 1, 2), dtype=np.float32):
        try:
            image = cv2.imread(image_path)
            image = np.transpose(image, transpose)
            image = np.array(image, dtype=dtype)
            image = image / 255
            image = np.array([image])

            return image

        except Exception as ex:
            exception_printer('Load image \'' + str(image_path) + '\' failed.')