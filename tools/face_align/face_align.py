import cv2
import numpy as np
from skimage import transform as trans

arcface_src = np.array([[38.2946, 51.6963],
                        [73.5318, 51.5014],
                        [56.0252, 71.7366],
                        [41.5493, 92.3655],
                        [70.7299, 92.2041]], dtype=np.float32)
arcface_src = np.expand_dims(arcface_src, axis=0)


# ================================================
# Estimate and normalize
#
# :param model_path
# :param initialize_image_path
# ================================================
def estimate_norm(landmarks, image_size=112):
    assert landmarks.shape == (5, 2)

    tform = trans.SimilarityTransform()
    lmk_tran = np.insert(landmarks, 2, values=np.ones(5), axis=1)

    min_M = []
    min_index = []
    min_error = float('inf')

    for i in np.arange(arcface_src.shape[0]):
        tform.estimate(landmarks, arcface_src[i])
        M = tform.params[0:2, :]
        results = np.dot(M, lmk_tran.T)
        results = results.T
        error = np.sum(np.sqrt(np.sum((results - arcface_src[i])**2, axis=1)))

        if error < min_error:
            min_error = error
            min_M = M
            min_index = i

    return min_M, min_index


# ================================================
# Face align
#
# :param image
# :param landmarks
# :param image_size
# ================================================
def face_align(image, landmarks, image_size=112):
    M, pose_index = estimate_norm(landmarks, image_size)
    warped = cv2.warpAffine(image, M, (image_size, image_size), borderValue=0.0)

    return warped