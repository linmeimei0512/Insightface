import os
import argparse
import random
import itertools


'''
=============================
Default
=============================
'''
datasets_images_dir = '../../../../Deep_Learning/InsightFace/Python/Dataset/faces_emore_mask/images2'
output_path = '../../../../Deep_Learning/InsightFace/Python/Dataset/faces_emore_mask/pairs2.txt'
pairs_num = 300
image_ext = 'jpg'


"""
=============================
Pairs.txt Generator
=============================
"""
class PairsGenerator:
    mac_hidden_file_name = '.DS_Store'

    """
    :param datasets_images_dir: is your images directory.
    :param output_pairs_path:   where is the pairs.txt that belongs to.
    :param pairs_num:           total pairs number
    :param image_ext:           is the image data extension for all of your image data.
    """
    def __init__(self, datasets_images_dir, output_pairs_path, pairs_num, image_ext):
        self.datasets_images_dir = datasets_images_dir
        self.output_pairs_path = output_pairs_path
        self.pairs_num = pairs_num
        self.image_ext = image_ext
        self.pairs_txt = None

        self.total_images_path = []
        self.total_similar_pairs_write_num = 0
        self.total_different_pairs_write_num = 0


    # ===============================
    # Start generate
    # ===============================
    def generate(self):
        self.labels_folders_list = os.listdir(self.datasets_images_dir)
        self.one_label_pairs_nun = self.pairs_num / len(self.labels_folders_list)
        print('One label need pairs num: ', self.one_label_pairs_nun)

        self.create_pairs_txt()

        # Use 10-fold cross-validation
        for i in range(10):
            self.write_similar()
            self.write_different()
        self.close_pairs_txt()

        print('\nSimilar pairs number: ' + str(self.total_similar_pairs_write_num))
        print('Different pairs number: ' + str(self.total_different_pairs_write_num))


    # ===============================
    # Create pairs.txt
    # ===============================
    def create_pairs_txt(self):
        if os.path.isfile(self.output_pairs_path):
            remove = input('\nThe pairs.txt is exist. Will you remove old pair.txt? (yes/no)')

            if remove.lower() == 'yes':
                os.remove(self.output_pairs_path)
                self.pairs_txt = open(self.output_pairs_path, 'a')
                self.pairs_txt.writelines('\n')
        else:
            self.pairs_txt = open(self.output_pairs_path, 'a')
            self.pairs_txt.writelines('\n')


    # ===============================
    # Close pairs.txt
    # ===============================
    def close_pairs_txt(self):
        if self.pairs_txt is not None:
            self.pairs_txt.close()


    # ===============================
    # Write similar to pairs.txt
    # ===============================
    def write_similar(self):
        if self.pairs_txt is None:
            return

        similar_pairs_write_num = 0

        # label count more than need pairs number
        if self.one_label_pairs_nun <= 1:
            while True:
                label_folder = random.choice(self.labels_folders_list)
                if label_folder == self.mac_hidden_file_name:
                    continue

                # get all images path with this label
                images_path_list = []
                for image in os.listdir(os.path.join(self.datasets_images_dir, label_folder)):
                    if image == self.mac_hidden_file_name:
                        continue

                    image_path = os.path.join(self.datasets_images_dir, label_folder, image)
                    images_path_list.append(image_path)

                # random choice 2 images with this label
                similar_images_path = random.choices(images_path_list, k=2)
                image_1_path = similar_images_path[0]
                image_2_path = similar_images_path[1]

                # write to pairs.txt
                self.pairs_txt.write(image_1_path + ' ' + image_2_path + ' 1\n')

                similar_pairs_write_num += 1
                self.total_similar_pairs_write_num += 1

                if similar_pairs_write_num == self.pairs_num:
                    break

        else:
            for label_folder in self.labels_folders_list:
                if label_folder == self.mac_hidden_file_name:
                    continue

                images_path_list = []
                for image in os.listdir(os.path.join(self.datasets_images_dir, label_folder)):
                    if image == self.mac_hidden_file_name:
                        continue

                    image_path = os.path.join(self.datasets_images_dir, label_folder, image)
                    images_path_list.append(image_path)
                    self.total_images_path.append(image_path)

                # random similar pairs
                similar_pairs_list = []
                for similar_pairs in itertools.combinations(images_path_list, 2):
                    similar_pairs_list.append(similar_pairs)
                    # total_similar_images_path_list.append(similar_pairs)

                for i in range(int(self.one_label_pairs_nun)):
                    similar_pairs = random.choice(similar_pairs_list)
                    image_1_path = similar_pairs[0]
                    image_2_path = similar_pairs[1]
                    self.pairs_txt.write(image_1_path + ' ' + image_2_path + ' 1\n')

                    similar_pairs_write_num += 1
                    self.total_similar_pairs_write_num += 1



        # print('Similar pairs number: ' + str(similar_pairs_write_num))


    # ===============================
    # Write different to pairs.txt
    # ===============================
    def write_different(self):
        if self.pairs_txt is None:
            return

        different_pairs_write_num = 0

        if len(self.total_images_path) == 0:
            self.get_total_images_path()

        random.shuffle(self.total_images_path)

        while different_pairs_write_num != self.pairs_num:
            different = random.choices(self.total_images_path, k=2)

            if self.get_label(different[0]) != self.get_label(different[1]):
                self.pairs_txt.write(different[0] + ' ' + different[1] + ' 0\n')
                different_pairs_write_num += 1
                self.total_different_pairs_write_num += 1

        # print('Different pairs number: ' + str(different_pairs_write_num))




    # ===============================
    # Get total images path
    # ===============================
    def get_total_images_path(self):
        for label_folder in self.labels_folders_list:
            if label_folder == self.mac_hidden_file_name:
                continue

            for image in os.listdir(os.path.join(self.datasets_images_dir, label_folder)):
                if image == self.mac_hidden_file_name:
                    continue

                image_path = os.path.join(self.datasets_images_dir, label_folder, image)
                self.total_images_path.append(image_path)


    # ===============================
    # Get image label
    # ===============================
    def get_label(self, image_path=str):
        image_path = image_path.replace('\\', '/')
        image_path = image_path.split('/')

        return image_path[len(image_path)-2]

'''
=============================
Main
=============================
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate pairs.txt')
    parser.add_argument('--datasets_images_dir', default=datasets_images_dir, help='is your dataset images directory.')
    parser.add_argument('--output_path', default=output_path, help='is where you want to save pairs.txt.')
    parser.add_argument('--pairs_num', default=pairs_num, type=int, help='is how many pairs that you want to create.')
    parser.add_argument('--image_ext', default=image_ext, help='is dataset images extension.')
    args = parser.parse_args()

    datasets_images_dir = args.datasets_images_dir
    output_path = args.output_path
    pairs_num = args.pairs_num
    image_ext = args.image_ext

    print('************** Generate pairs.txt **************')
    print('datasets images dir: ' + str(datasets_images_dir))
    print('output path: ' + str(output_path))
    print('pairs num: ' + str(pairs_num))
    print('images extension: ' + str(image_ext))

    pairs_generator = PairsGenerator(datasets_images_dir, output_path, pairs_num, image_ext)
    pairs_generator.generate()