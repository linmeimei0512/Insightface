import argparse
import os
import time
from easydict import EasyDict as edict
import numpy as np
import mxnet as mx

'''
=============================
Default
=============================
'''
datasets_images_dir = '../../../../Deep Learning/InsightFace/Python/Dataset/faces_emore_mask/images'
output_path = '../../../../Deep Learning/InsightFace/Python/Dataset/faces_emore_mask/train.rec'
image_size = '112,112'
image_ext = 'jpg'


'''
=============================
List Generator
=============================
'''
class ListGenerator:
    """
    :param datasets_images_dir: is your images directory.
    :param list_output_path:    where is the .lst that belongs to.
    :param image_ext:           is the image data extension for all of your image data.
    """
    def __init__(self, datasets_images_dir, list_output_path, image_ext):
        self.datasets_images_dir = datasets_images_dir
        self.list_output_path = list_output_path
        self.image_ext = image_ext
        self.lst_file = None


    # ===============================
    # Create .lst file
    # ===============================
    def create_lst_file(self):
        if os.path.isfile(self.list_output_path):
            os.remove(self.list_output_path)

        self.lst_file = open(self.list_output_path, 'a')

    # ===============================
    # Close .lst file
    # ===============================
    def close_lst_file(self):
        if self.lst_file is not None:
            self.lst_file.close()

    # ===============================
    # Generate .lst
    # ===============================
    def generate(self):
        print('Generate ' + str(self.list_output_path) + ' ...')
        start_time = time.time()
        self.create_lst_file()
        cnt = 0
        labels = []

        for label in os.listdir(self.datasets_images_dir):
            if label == '.DS_Store':
                continue
            labels.append(label)
        labels = sorted(labels)

        for label in labels:
            for image in os.listdir(self.datasets_images_dir + '/' + label):
                if image == '.DS_Store':
                    continue
                self.lst_file.write(str(1) + '\t' + datasets_images_dir + '/' + label + '/' + image + '\t' + str(cnt) + '\n')
            cnt += 1

        self.close_lst_file()
        print('Generate finish. Time cost: ' + str(time.time() - start_time) + ' sec.')

'''
=============================
REC Generator
=============================
'''
class RECGenerator:
    """
    :param datasets_images_dir: is your images directory.
    :param list_path:           where is the .lst.
    :param output_rec_path:     where is the .rec belongs to.
    :param image_size:          is the image data size for all of your image data.
    :param color:               specify the color mode of the loaded image.
    """
    def __init__(self, datasets_images_dir, lst_path, output_rec_path, image_size, color):
        self.datasets_images_dir = datasets_images_dir
        self.lst_path = lst_path
        self.output_rec_path = output_rec_path
        self.output_idx_path = output_rec_path.replace('.rec', '.idx')
        self.image_size = image_size
        self.color = color
        return

    # ===============================
    # Generate
    # ===============================
    def generate(self):
        print('\nCreating .rec file from ' + str(self.lst_path))

        image_list = self.read_list(self.lst_path)

        try:
            import Queue as queue
        except ImportError:
            import queue
        q_out = queue.Queue()
        record = mx.recordio.MXIndexedRecordIO(
            self.output_idx_path,
            self.output_rec_path, 'w')
        cnt = 0
        pre_time = time.time()

        for i, item in enumerate(image_list):
            self.image_encode(i, item, q_out)
            if q_out.empty():
                continue
            _, s, item = q_out.get()

            record.write_idx(item[0], s)
            if cnt % 1000 == 0:
                cur_time = time.time()
                print('time:', cur_time - pre_time, ' count:', cnt)
                pre_time = cur_time
            cnt += 1

        print('Generate finish')
        print('.rec saved as ' + self.output_rec_path)
        print('.idx saved as ' + self.output_idx_path)

    # ===============================
    # Read list
    # ===============================
    def read_list(self, path_in):
        with open(path_in) as fin:
            identities = []
            last = [-1, -1]
            _id = 1
            while True:
                line = fin.readline()
                if not line:
                    break
                item = edict()
                item.flag = 0
                item.image_path, label, item.bbox, item.landmark, item.aligned = self.parse_lst_line(line)
                if not item.aligned and item.landmark is None:
                    # print('ignore line', line)
                    continue
                item.id = _id
                item.label = [label, item.aligned]
                yield item
                if label != last[0]:
                    if last[1] >= 0:
                        identities.append((last[1], _id))
                    last[0] = label
                    last[1] = _id
                _id += 1
            identities.append((last[1], _id))
            item = edict()
            item.flag = 2
            item.id = 0
            item.label = [float(_id), float(_id + len(identities))]
            yield item
            for identity in identities:
                item = edict()
                item.flag = 2
                item.id = _id
                _id += 1
                item.label = [float(identity[0]), float(identity[1])]
                yield item

    # ===============================
    # Parse lst line
    # ===============================
    def parse_lst_line(self, line):
        vec = line.strip().split("\t")
        assert len(vec) >= 3
        aligned = int(vec[0])
        image_path = vec[1]
        label = int(vec[2])
        bbox = None
        landmark = None
        # print(vec)
        if len(vec) > 3:
            bbox = np.zeros((4,), dtype=np.int32)
            for i in range(3, 7):
                bbox[i - 3] = int(vec[i])
            landmark = None
            if len(vec) > 7:
                _l = []
                for i in range(7, 17):
                    _l.append(float(vec[i]))
                landmark = np.array(_l).reshape((2, 5)).T
        # print(aligned)
        return image_path, label, bbox, landmark, aligned

    # ===============================
    # Image encode with MXNet
    # ===============================
    def image_encode(self, i, item, q_out):
        oitem = [item.id]
        # print('flag', item.flag)
        if item.flag == 0:
            fullpath = item.image_path
            header = mx.recordio.IRHeader(item.flag, item.label, item.id, 0)
            # print('write', item.flag, item.id, item.label)
            if item.aligned:
                with open(fullpath, 'rb') as fin:
                    img = fin.read()
                s = mx.recordio.pack(header, img)
                q_out.put((i, s, oitem))
            # else:
            #     img = cv2.imread(fullpath, color)
            #     assert item.landmark is not None
            #     img = face_align.norm_crop(img, item.landmark)
            #     s = mx.recordio.pack_img(header,
            #                              img,
            #                              quality=args.quality,
            #                              img_fmt=encoding)
            #     q_out.put((i, s, oitem))
        else:
            header = mx.recordio.IRHeader(item.flag, item.label, item.id, 0)
            # print('write', item.flag, item.id, item.label)
            s = mx.recordio.pack(header, b'')
            q_out.put((i, s, oitem))


# ===============================
# Parse
# ===============================
def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Create an image list or make a record database by reading from an image list')
    parser.add_argument('--datasets_images_dir', default= datasets_images_dir, help='is your dataset images directory.')
    parser.add_argument('--output_path', default=output_path, help='is where you want to save .rec')
    parser.add_argument('--image_size', default=image_size, help='is dataset image size.')
    parser.add_argument('--image_ext', default=image_ext, help='is dataset images extension.')
    parser.add_argument('--color',
                        type=int,
                        default=1,
                        choices=[-1, 0, 1],
                        help='specify the color mode of the loaded image. ' +
                             '1: Loads a color image. Any transparency of image will be neglected. It is the default flag. ' +
                             '0: Loads image in grayscale mode. ' +
                             '-1:Loads image as such including alpha channel.')

    args = parser.parse_args()
    return args

# ===============================
# Find list file (.lst)
# ===============================
def find_list(rec_output_path:str):
    rec_output_path = rec_output_path.replace('\\', '/')
    rec_output_path = rec_output_path.split('/')

    rec_name = rec_output_path[len(rec_output_path)-1]
    rec_name = rec_name.split('.')
    rec_name = rec_name[0]

    list_path = ''
    for i in range(0, len(rec_output_path)-1):
        list_path += (rec_output_path[i] + '/')

    list_path += (rec_name + '.lst')

    if os.path.isfile(list_path):
        return True, list_path
    else:
        return False, list_path


'''
=============================
Main
=============================
'''
if __name__ == '__main__':
    args = parse_args()

    datasets_images_dir = args.datasets_images_dir
    output_path = args.output_path
    image_size = [int(x) for x in args.image_size.split(',')]
    image_ext = args.image_ext
    color = args.color

    print('************** Generate .rec **************')
    print('datasets images dir: ' + str(datasets_images_dir))
    print('output path: ' + str(output_path))
    print('image size: ' + str(image_size))
    print('images extension: ' + str(image_ext))
    print('image color: ' + str(color))

    # Check .lst file
    generate_list = False
    list_is_exist, list_path = find_list(rec_output_path=output_path)
    if list_is_exist:
        remove = input('\nThe %s is exist. Will you remove old? (yes/no)' %list_path)
        if remove.lower() == 'yes':
            generate_list = True
    else:
        print('\n' + list_path + ' is not exist!')
        generate_list = True

    if generate_list:
        list_generator = ListGenerator(datasets_images_dir=datasets_images_dir,
                                       list_output_path=list_path,
                                       image_ext=image_ext)
        list_generator.generate()

    # Generate .rec and .idx
    rec_generator = RECGenerator(datasets_images_dir=datasets_images_dir,
                                 lst_path=list_path,
                                 output_rec_path=output_path,
                                 image_size=image_size,
                                 color=color)
    rec_generator.generate()
