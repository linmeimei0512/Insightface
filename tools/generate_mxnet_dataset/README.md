
# Generate MXNet Use Dataset

## Recent Update

**`2021-05-19`**: Fix generate pairs.txt with generate_pairs.py. Use 10-fold cross-validation. See [here](generate_pairs.py).

**`2021-05-06`**: First commit.



## Content
[Generate MXNet Use Dataset]()
- [Generate .rec](#generate-.rec)
- [Generate pairs.txt](#generate-pairs.txt)
- [Generate bin](#generate-.bin)



## Set Up

| Python                    | 3.6       | 
| :---                      | :---      |

#####CPU Version
| Package                   | Version   | 
| :---                      | :---      |
| mxnet                     | 1.7.0.post2 |
| numpy                     | 1.19.5    |
| esaydict                  | 1.9       |
| psutil                    | 5.8.0     |

#####GPU Version (CUDA 10)
| Package                   | Version   | 
| :---                      | :---      |
| mxnet-cu100               | 1.5.0     |
| numpy                     | 1.19.5    |
| esaydict                  | 1.9       |
| psutil                    | 5.8.0     |


### Create Python 
```
conda create -n Python3.6 python=3.6
```
 
### Install Python Package
#####CPU Version
[requirements.txt](requirements.txt)
```
pip install -r requirements.txt
```

#####GPU Version (CUDA 10)
[requirements-gpu.txt](requirements-gpu.txt)
```
pip install -r requirements-gpu.txt
```



## Generate .rec

```
usage: generate_rec.py [-h] [--datasets_images_dir DATASETS_IMAGES_DIR]
                       [--output_path OUTPUT_PATH] [--image_size IMAGE_SIZE]
                       [--image_ext IMAGE_EXT] [--color {-1,0,1}]

Create an image list or make a record database by reading from an image list

optional arguments:
  -h, --help            show this help message and exit
  --datasets_images_dir DATASETS_IMAGES_DIR
                        is your dataset images directory. 
                        (default:../../../../Deep Learning/InsightFace/Python/Dataset/faces_emore_mask/images)
  --output_path OUTPUT_PATH
                        is where you want to save .rec 
                        (default:../../../../Deep Learning/InsightFace/Python/Dataset/faces_emore_mask/train.rec)
  --image_size IMAGE_SIZE
                        is dataset image size. 
                        (default: 112,112)
  --image_ext IMAGE_EXT
                        is dataset images extension. 
                        (default: jpg)
  --color {-1,0,1}      specify the color mode of the loaded image. 
                        1: Loads a color image. Any transparency of image will be neglected. It is the default flag. 
                        0: Loads image in grayscale mode. 
                        -1:Loads image as such including alpha channel. (default: 1)
```

```
python generate_rec.py --datasets_images_dir "../../../../Deep_Learning/InsightFace/Python/Dataset/faces_emore_mask/images" --output_path "../../../../Deep_Learning/InsightFace/Python/Dataset/faces_emore_mask/train.rec"
```


## Generate pairs.txt

```
usage: generate_pairs.py [-h] [--datasets_images_dir DATASETS_IMAGES_DIR] [--output_path OUTPUT_PATH]
                         [--pairs_num PAIRS_NUM] [--image_ext IMAGE_EXT]

Generate pairs.txt

optional arguments:
  -h, --help            show this help message and exit
  --datasets_images_dir DATASETS_IMAGES_DIR
                        is your dataset images directory.
  --output_path OUTPUT_PATH
                        is where you want to save pairs.txt.
  --pairs_num PAIRS_NUM
                        is how many pairs that you want to create.
  --image_ext IMAGE_EXT
                        is dataset images extension.
```

```
python generate_pairs.py --datasets_images_dir "../../../../Deep_Learning/InsightFace/Python/Dataset/faces_emore_mask/images" --output_path "../../../../Deep_Learning/InsightFace/Python/Dataset/faces_emore_mask/pairs.txt" --pairs_num 3000 --image_ext jpg
```


## Generate .bin

```
usage: generate_bin.py [-h] [--pairs_txt_path PAIRS_TXT_PATH]
                       [--output_bin_path OUTPUT_BIN_PATH]

Generate pin

optional arguments:
  -h, --help            show this help message and exit````
  --pairs_txt_path PAIRS_TXT_PATH
                        where is the pairs.txt.
  --output_bin_path OUTPUT_BIN_PATH
                        where to save the output .bin.
```

```
python generate_bin.py --pairs_txt_path "../../../../Deep_Learning/InsightFace/Python/Dataset/faces_emore_mask/pairs.txt" --output_bin_path "../../../../Deep_Learning/InsightFace/Python/Dataset/faces_emore_mask/faces_emore_mask.bin"
```