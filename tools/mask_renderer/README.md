
# Mask Renderer

## Recent Update

**`2021-05-26`**: First commit.




## Content
[Mask Renderer]()
- [Set up environment](#set-up)
- [Download model](#download-model)
- [Render mask](#render-mask)



## Set Up
[Set up environment](#set-up)
- [Create python environment](#create-python-environment)
- [Install python package](#install-python-package)



## Create Python Environment

| Python                    | 3.6       | 
| :---                      | :---      |

```
conda create -n insightface python=3.6
```


## Install Python Package
#### CPU Version

| Package                   | Version   | 
| :---                      | :---      |
| tqdm                      | latest    |
| numpy                     | latest    |
| requests                  | latest    |
| mxnet                     | latest    |

[requirements.txt](requirements.txt)
```
pip install -r requirements.txt
```

#### GPU Version (CUDA 10.1)

| Package                   | Version   | 
| :---                      | :---      |
| tqdm                      | latest    |
| numpy                     | latest    |
| requests                  | latest    |
| mxnet-cu101               | 1.5.0     |

[requirements-gpu.txt](requirements-gpu.txt)
```
pip install -r requirements-gpu.txt
```


## Download Model
Download BFM and 3d68 model to assets_mask folder.

[Model-Zoo](../../model_zoo/README.md)



## Render Mask

```
usage: mask_renderer.py [-h] [--mask_image_path MASK_IMAGE_PATH]
                        [--image_path IMAGE_PATH] [--use_face3d]
                        [--save_image_path SAVE_IMAGE_PATH]
                        [--show_image SHOW_IMAGE]

Mask Renderer

optional arguments:
  -h, --help            show this help message and exit
  --mask_image_path MASK_IMAGE_PATH
                        where is the mask photo.
  --image_path IMAGE_PATH
                        where is the photo that need render mask.
  --use_face3d          use face3d.
  --save_image_path SAVE_IMAGE_PATH
                        where to save the photo that rendered mask.
  --show_image SHOW_IMAGE
                        show the photo that rendered mask. default: True.
```


### Use OpenCV

```
python mask_renderer.py --mask_image_path mask_image/mask_04.png --image_path ../../images/Tom_Hanks_01.jpg --save_image_path ../../images/Tom_Hanks_01(mask).jpg
```

### Use face3d

```
python mask_renderer.py --mask_image_path mask_image/mask1.jpg --image_path ../../images/Tom_Hanks_01.jpg --use_face3d --save_image_path ../../images/Tom_Hanks_01(mask).jpg
```