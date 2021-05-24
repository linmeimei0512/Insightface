import sys
import os
import cv2
import numpy as np

'''
########################################
Mask Renderer
########################################
'''
class MaskRenderer:
    mask_image_path = os.path.join(os.path.dirname(__file__), 'mask_image/mask_04.png')
    mask_image = None

    def __init__(self, mask_image_path=None):
        if mask_image_path is not None:
            self.mask_image_path = mask_image_path

        print('\n********** Mask Renderer **********')
        print('mask image path: ' + str(self.mask_image_path))

        self.init_mask_image()
        print('***********************************')


    # ================================================
    # Initialize mask image
    # ================================================
    def init_mask_image(self):
        if not os.path.exists(self.mask_image_path):
            print('Initialize mask image failed. Mask image \'' + str(self.mask_image_path) + '\' is not exist.')
            return

        image = cv2.imread(self.mask_image_path, cv2.IMREAD_UNCHANGED)
        image = cv2.resize(image, (90, 55))

        add_alpha = np.zeros((112, 112, 4), dtype=np.uint8)
        add_alpha[57:112, 11:101] = image

        self.mask_image = add_alpha
        print('Initialize mask image success.')


    # ================================================
    # Render mask
    # ================================================
    def render(self, image):
        if self.mask_image is None:
            return

        image = image.copy()

        b, g, r, a = cv2.split(self.mask_image)
        foreground = cv2.merge((b, g, r))

        alpha = cv2.merge((a, a, a))
        alpha = alpha.astype(float) / 255

        foreground = foreground.astype(float)
        background = image.astype(float)

        foreground = cv2.multiply(alpha, foreground)
        background = cv2.multiply(1 - alpha, background)

        outImage = foreground + background
        outImage = outImage.astype(np.uint8)

        return outImage