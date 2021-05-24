import sys
import os
import time
import tensorflow as tf
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.exception_printer import exception_printer

'''
########################################
MTCNN - Tensorflow
########################################
'''
class MTCNN:
    initialize_image_path = os.path.join(os.path.dirname(__file__), 'initialize_image.jpg')

    ################
    #### factor
    ################
    no_mask_factor = 0.709
    mask_factor = 0.97

    ################
    #### Threshold
    ################
    no_mask_thresholds = [0.6, 0.7, 0.7]
    mask_thresholds = [0.4, 0.2, 0.2]


    # ================================================
    # Constructor
    #
    # :param model_path
    # :param initialize_image_path
    # :param min_size
    # :param factor
    # :param threshold
    # ================================================
    def __init__(self, model_path, initialize_image_path=None, min_size=100, factor=0.709, thresholds=[0.6, 0.7, 0.7]):
        if initialize_image_path is not None:
            self.initialize_image_path = initialize_image_path
        self.min_size = min_size
        self.factor = factor
        self.thresholds = thresholds

        print('\n********** MTCNN Tensorflow **********')
        print('model path: ' + str(model_path))
        print('min size: ' + str(min_size))
        print('factor: ' + str(factor))
        print('threshold: ' + str(thresholds))

        self.load_mtcnn_model(model_path)

        print('**************************************')


    # ================================================
    # Load MTCNN model
    # ================================================
    def load_mtcnn_model(self, model_path):
        print('\nStarting load MTCNN tensorflow model \'' + str(model_path) + '\'...')
        start_time = time.time()

        try:
            graph = tf.Graph()
            with graph.as_default():
                with open(model_path, 'rb') as f:
                    graph_def = tf.compat.v1.GraphDef.FromString(f.read())
                    tf.compat.v1.import_graph_def(graph_def=graph_def, name='')

            self.graph = graph

            config = tf.compat.v1.ConfigProto(allow_soft_placement=True, intra_op_parallelism_threads=4, inter_op_parallelism_threads=4)
            config.gpu_options.allow_growth = True

            self.sess = tf.compat.v1.Session(graph=graph, config=config)

            # initialize mtcnn model
            self.init_mtcnn_model()

        except Exception as ex:
            exception_printer('Load MTCNN model failed. ')

        print('Load MTCNN tensorflow model success. Cost time: ' + str(time.time() - start_time))


    # ================================================
    # Initialize MTCNN model
    #
    # :param initialize image path
    # ================================================
    def init_mtcnn_model(self):
        if self.initialize_image_path is None:
            print('Initialize MTCNN model failed. Initialize image path is None.')
            return

        if not os.path.exists(self.initialize_image_path):
            print('Initialize MTCNN model failed. Initialize image path \'' + str(self.initialize_image_path) + '\' is not exist.')
            return

        initialize_image = cv2.imread(self.initialize_image_path)
        self.detect(initialize_image, False, 0)
        print('Initialize MTCNN model success.')


    # ================================================
    # Detect Face
    #
    # :param img
    # :param have_mask
    # :param score_threshold
    #
    # return boxes
    #        scores
    #        points
    # ================================================
    def detect(self, img, have_mask, score_threshold):
        if have_mask:
            self.factor = self.mask_factor
            self.thresholds = self.mask_thresholds
        else:
            self.factor = self.no_mask_factor
            self.thresholds = self.no_mask_thresholds

        feeds = {
            self.graph.get_operation_by_name('input').outputs[0]: img,
            self.graph.get_operation_by_name('min_size').outputs[0]: self.min_size,
            self.graph.get_operation_by_name('thresholds').outputs[0]: self.thresholds,
            self.graph.get_operation_by_name('factor').outputs[0]: self.factor
        }
        fetches = [self.graph.get_operation_by_name('prob').outputs[0],
                   self.graph.get_operation_by_name('landmarks').outputs[0],
                   self.graph.get_operation_by_name('box').outputs[0]]

        temp_scores, temp_points, temp_boxes = self.sess.run(fetches, feeds)

        boxes = []
        scores = []
        points = []

        for box, score, point in zip(temp_boxes, temp_scores, temp_points):
            if score >= score_threshold:
                boxes.append(box)
                scores.append(score)
                points.append(point)

        return boxes, scores, points