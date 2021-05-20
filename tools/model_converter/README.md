
# Insightface PyTorch To Tensorflow Lite (.tflite)

## Recent Update

**`2021-05-20`**: First commit.



## Content
[Insightface PyTorch model (.pth) convert to Tensorflow Lite model (.tflite)]()
- [PyTorch to Tensorflow Lite](#pytorch-to-tensorflow-lite)


## Set Up

| Python                    | 3.6       | 
| :---                      | :---      |

| PyTorch                   | 1.7       | 
| :---                      | :---      |

#### CPU Version
| Package                   | Version   | 
| :---                      | :---      |
| traceback2                | 1.4.0 |
| numpy                     | 1.19.5    |
| opencv-python             | 4.5.2.52  |
| torchsummary              | 1.5.1     |
| pkg-resources1            | 0.0.7     |
| scipy                     | 1.5.4     |
| onnx-tf                   | 1.6.0     |
| onnxruntime               | 1.7.0     |
| tensorflow                | 2.4.1     |
| tensorflow_addons         | 0.12.1    |

#### GPU Version (CUDA 10.1)
| Package                   | Version   | 
| :---                      | :---      |
| traceback2                | 1.4.0 |
| numpy                     | 1.19.5    |
| opencv-python             | 4.5.2.52  |
| torchsummary              | 1.5.1     |
| pkg-resources1            | 0.0.7     |
| scipy                     | 1.5.4     |
| onnx-tf                   | 1.6.0     |
| onnxruntime               | 1.7.0     |
| tensorflow-gpu            | 2.4.1     |
| tensorflow_addons         | 0.12.1    |

[Set up environment]()
- [Create Python Environment](#create-python-environment)
- [Install PyTorch](#install-pytorch)
- [Install Python Package](#install-python-package)


## Create Python Environment
Python 3.6
```
conda create -n insightface python=3.6
```


## Install PyTorch
### v1.7.1 - Linux and Windows  
```shell
# CUDA 11.0
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 10.2
pip install torch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2

# CUDA 10.1
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 9.2
pip install torch==1.7.1+cu92 torchvision==0.8.2+cu92 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# CPU only
pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```
More details see [PyTorch Website](https://pytorch.org/get-started/previous-versions/) in docs.



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


## PyTorch To Tensorflow Lite

```
usage: pytorch_to_tflite.py [-h] [--pytorch_model_path PYTORCH_MODEL_PATH]
                            [--pytorch_weight_path PYTORCH_WEIGHT_PATH]
                            [--input_shape INPUT_SHAPE]
                            [--input_names INPUT_NAMES]
                            [--output_names OUTPUT_NAMES]
                            [--tensorflow_lite_model_output_path TENSORFLOW_LITE_MODEL_OUTPUT_PATH]

PyTorch model convert to tflite model

optional arguments:
  -h, --help            show this help message and exit
  --pytorch_model_path PYTORCH_MODEL_PATH
                        where is the pytorch model.
  --pytorch_weight_path PYTORCH_WEIGHT_PATH
                        where is the pytorch weight.
  --input_shape INPUT_SHAPE
                        input shape for pytorch model. ex. 3,112,112
  --input_names INPUT_NAMES
                        the input names to use for onnx model. ex. input
  --output_names OUTPUT_NAMES
                        the output name to use for onnx model. ex. output
  --tensorflow_lite_model_output_path TENSORFLOW_LITE_MODEL_OUTPUT_PATH
                        where is save to tflite model.

```

```
python pytorch_to_tflite.py --pytorch_model_path "../../../../Deep_Learning/InsightFace/Python/Models/PyTorch/emore_mask_resnet_r18/6_backbone.pth" --input_shape 3,112,112 --input_names input --output_names output --tensorflow_lite_model_output_path "../../../../Deep_Learning/InsightFace/Python/Models/PyTorch/emore_mask_resnet_r18/emore_mask_r18.tflite"  
```