import sys
import cv2
import os
import numpy as np
import argparse
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from detection.mtcnn.mtcnn_tensorflow import MTCNN
from tools.face_align import face_align
from tools.mask_renderer.mask_renderer import MaskRenderer


mtcnn_model_path = '../model_zoo/mtcnn/mtcnn.pb'


# ================================================
# Initialize MTCNN
#
# return mtcnn model
# ================================================
def init_mtcnn_model():
    mtcnn = MTCNN(model_path=mtcnn_model_path)

    return mtcnn


# ================================================
# Detect face
#
# :param mtcnn
# :param image_path
# :param have_mask
# :param score_threshold
#
# return boxes
#        scores
#        points
# ================================================
def face_detect(mtcnn, image, have_mask=False, score_threshold=0.9):
    boxes, scores, points = mtcnn.detect(image, have_mask=have_mask, score_threshold=score_threshold)

    return boxes, scores, points


# ================================================
# Crop face
#
# :param image
# :param landmark
#
# return face_image
# ================================================
def crop_face(image, landmark):
    landmark = np.array([landmark[5], landmark[6], landmark[7], landmark[8], landmark[9], landmark[0], landmark[1], landmark[2], landmark[3], landmark[4]])
    landmark = landmark.reshape((2, 5)).T
    face_image = face_align.face_align(image, landmark)

    return face_image


# ================================================
# Initialize mask renderer
#
# return mask_renderer
# ================================================
def init_mask_renderer():
    mask_renderer = MaskRenderer()

    return mask_renderer



'''
=============================
Main
=============================
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face detection')
    parser.add_argument('--use_model', type=str, default='mtcnn', help='detect face use mtcnn or retinaface.')
    parser.add_argument('--mask_renderer', type=bool, default=False, help='mask renderer')
    parser.add_argument('--image_path', type=str, default='../images/initialize_image.jpg', help='detect face image path')
    parser.add_argument('--save_face_photo', type=bool, default=False, help='save detected faces.')
    parser.add_argument('--show_face_photo', type=bool, default=False, help='show detected faces.')
    args = parser.parse_args()

    print('\n********** Face Detection **********')
    print('use model: ', args.use_model)
    print('mask renderer: ', args.mask_renderer)
    print('image path: ', args.image_path)
    print('save face photo: ', args.save_face_photo)
    print('show face photo: ', args.show_face_photo)
    print('************************************')

    if args.use_model:
        mtcnn = init_mtcnn_model()

    if args.mask_renderer:
        mask_renderer = init_mask_renderer()

    print('\nDetect faces from ' + str(args.image_path) + '...')
    if not os.path.exists(args.image_path):
        print(str(args.image_path) + ' is not exist.')
        sys.exit()
    image = cv2.imread(args.image_path)

    start_time = time.time()
    boxes, scores, points = face_detect(mtcnn, image)
    print(str(len(boxes)) + ' faces detected. Cost time: ' + str(time.time() - start_time))

    i = 0
    for point in points:
        face_image = crop_face(image, point)
        if args.show_face_photo:
            cv2.imshow(str(i), face_image)
        if args.save_face_photo:
            cv2.imwrite('../images/face_image_' + str(i) + '.jpg', face_image, [cv2.IMWRITE_JPEG_QUALITY, 100])

        if args.mask_renderer:
            face_mask_image = mask_renderer.render(face_image)
            if args.show_face_photo:
                cv2.imshow(str(i) + '_mask', face_mask_image)
            if args.save_face_photo:
                cv2.imwrite('../images/face_mask_image_' + str(i) + '.jpg', face_mask_image, [cv2.IMWRITE_JPEG_QUALITY, 100])

        i += 1

    cv2.waitKey(0)