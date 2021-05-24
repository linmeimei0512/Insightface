
# Face Detection

## Recent Update

**`2021-05-24`**: First commit.



## Content
[Face Detection]()
- [MTCNN]()



## Set Up

#### CPU Version
| Package                   | Version   | 
| :---                      | :---      |
| opencv-python             | 4.5.2.52  |
| tensorflow                | 2.4.1     |
| scikit-image              | latest    |

#### GPU Version (CUDA 10.1)
| Package                   | Version   | 
| :---                      | :---      |
| opencv-python             | 4.5.2.52  |
| tensorflow-gpu            | 2.4.1     |
| scikit-image              | latest    |


[Set up environment]()
- [Create Python Environment](#create-python-environment)
- [Install Python Package](#install-python-package)
- [Download MTCNN Model](#download-model)



## Create Python Environment
Python 3.6
```
conda create -n insightface python=3.6
```



## Install Python Package
#### CPU Version
[requirements.txt](requirements.txt)
```
pip install -r requirements.txt
```

#### GPU Version (CUDA 10.1)
[requirements-gpu.txt](requirements-gpu.txt)
```
pip install -r requirements-gpu.txt
```



## Download Model
[Model-Zoo](../model_zoo/README.md)

Copy model to ./model_zoo folder.



## Detection

```
usage: face_detection.py [-h] [--use_model USE_MODEL]
                         [--mask_renderer MASK_RENDERER]
                         [--image_path IMAGE_PATH]
                         [--save_face_photo SAVE_FACE_PHOTO]
                         [--show_face_photo SHOW_FACE_PHOTO]

Face detection

optional arguments:
  -h, --help            show this help message and exit
  --use_model USE_MODEL
                        detect face use mtcnn or retinaface. default: mtcnn
  --mask_renderer MASK_RENDERER
                        mask renderer. default: False
  --image_path IMAGE_PATH
                        detect face image path. default: ../images/initialize_image.jpg
  --save_face_photo SAVE_FACE_PHOTO
                        save detected faces. default: False
  --show_face_photo SHOW_FACE_PHOTO
                        show detected faces. default: False
```

```
python face_detection.py --use_model mtcnn
```