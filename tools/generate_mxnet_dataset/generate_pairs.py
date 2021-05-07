import os
import random
import itertools
import argparse

'''
=============================
Default
=============================
'''
datasets_images_dir = '../../../../Deep_Learning/InsightFace/Python/Dataset/faces_emore_mask/images'
output_path = '../../../../Deep_Learning/InsightFace/Python/Dataset/faces_emore_mask/pairs.txt'
pairs_num = 3000
image_ext = 'jpg'

pairs_txt = None
total_images_path_list = []
diff_list = []

"""
=============================
Pairs.txt Generator
=============================
"""
class PairsGenerator:
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

        self.create_pairs_txt()
        self.write_similar()
        self.write_different()
        self.close_pairs_txt()


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

        self.total_images_path = []
        self.similar_pairs_write_num = 0
        labels_folders = os.listdir(self.datasets_images_dir)
        label_pairs_nun = int(int(self.pairs_num) / len(labels_folders))
        print('Label pairs num: ', label_pairs_nun)

        for label_folder in labels_folders:
            if label_folder == '.DS_Store':
                continue

            # get all images path with label
            images_path_list = []
            for image in os.listdir(os.path.join(self.datasets_images_dir, label_folder)):
                if image == '.DS_Store':
                    continue

                image_path = os.path.join(self.datasets_images_dir, label_folder, image)
                images_path_list.append(image_path)
                self.total_images_path.append(image_path)


            # random similar pairs
            similar_pairs_list = []
            for similar_pairs in itertools.combinations(images_path_list, 2):
                similar_pairs_list.append(similar_pairs)

            for i in range(label_pairs_nun):
                similar_pairs = random.choice(similar_pairs_list)
                image_1_path = similar_pairs[0]
                image_2_path = similar_pairs[1]
                self.pairs_txt.write(image_1_path + ' ' + image_2_path + ' 1\n')
                self.similar_pairs_write_num = self.similar_pairs_write_num + 1

        print('Similar pairs number: ' + str(self.similar_pairs_write_num))


    # ===============================
    # Write different to pairs.txt
    # ===============================
    def write_different(self):
        self.different_pairs_write_num = 0
        random.shuffle(self.total_images_path)

        while self.different_pairs_write_num != self.similar_pairs_write_num:
            different = random.choices(self.total_images_path, k=2)

            if self.get_label(different[0]) != self.get_label(different[1]):
                self.pairs_txt.write(different[0] + ' ' + different[1] + ' 0\n')
                self.different_pairs_write_num = self.different_pairs_write_num + 1

        print('Different pairs number: ' + str(self.different_pairs_write_num))


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

    images_folders = os.listdir(datasets_images_dir)


    print('************** Generate pairs.txt **************')
    print('datasets images dir: ' + str(datasets_images_dir))
    print('output path: ' + str(output_path))
    print('pairs num: ' + str(pairs_num))
    print('images extension: ' + str(image_ext))

    pairs_generator = PairsGenerator(datasets_images_dir, output_path, pairs_num, image_ext)