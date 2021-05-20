import numpy as np
from enum import Enum
from scipy.spatial.distance import cosine

'''
Compare Distance Type
'''
class CompareDistanceType(Enum):
    Euclidean = 'euclidean'
    Cosine = 'cosine'


'''
########################################
Compare Util
########################################
'''
class Compare_Util:
    def __init__(self, debug):
        self.debug = debug

    '''
    Compare Feature
    '''
    def compare_feature(self, compare_distance_type, feature, check_feature, euclidean_distance_threshold, cosine_distance_threshold, similarity_threshold):

        # Euclidean distance
        euclidean_distance = np.sum(np.square(check_feature - feature))
        euclidean_distance = float(str(euclidean_distance)[:str(euclidean_distance).find('.') + 4])

        # Cosine distance
        cosine_distance = 1 - cosine(feature, check_feature)

        distance_pass = False
        if compare_distance_type == CompareDistanceType.Euclidean:
            if euclidean_distance <= euclidean_distance_threshold:
                distance_pass = True

        else:
            if cosine_distance >= cosine_distance_threshold:
                distance_pass = True

        # Similarity
        similarity = np.dot(check_feature, feature.T)
        similarity = float(str(similarity)[:str(similarity).find('.') + 4])
        similarity_pass = False
        if similarity <= similarity_threshold:
            similarity_pass = True

        return distance_pass, euclidean_distance, cosine_distance, similarity_pass, similarity

    '''
    Show Compare Data
    '''
    def show_compare_data(self, compare_datas):
        print('Data size: ' + str(len(compare_datas)))

        for compare_data in compare_datas:
            print('%-40s' % (compare_data.employee_data.name), end='')
        print('')

        for compare_data in compare_datas:
            print('%-40s' % (' <No mask>'), end='')
        print('')

        for compare_data in compare_datas:
            print('%-40s' % ('   euclidean distance: ' + str(compare_data.euclidean_distance)), end='')
        print('')

        for compare_data in compare_datas:
            print('%-40s' % ('   cosine distance: ' + str(compare_data.cosine_distance)), end='')
        print('')

        for compare_data in compare_datas:
            print('%-40s' % (' <Mask>'), end='')
        print('')

        for compare_data in compare_datas:
            print('%-40s' % ('   euclidean distance: ' + str(compare_data.mask_euclidean_distance)), end='')
        print('')

        for compare_data in compare_datas:
            print('%-40s' % ('   cosine distance: ' + str(compare_data.mask_cosine_distance)), end='')
        print('')
