
# Face Recognition - ONNX

## Recent Update

**`2022-01-13`**: First commit.



## Content
[Face Recognition - ONNX]()
- [Set up environment](#set-up)
- [Verify model](#verify)


## Create Python Environment

| Python                    | 3.6       | 
| :---                      | :---      |

```
conda create -n insightface-onnx python=3.6
```

## Install PyTorch

| PyTorch                   | 1.7.1     | 
| :---                      | :---      |

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

| Package                   | Version   | 
| :---                      | :---      |
| easydict                  | 1.9       |
| numpy                     | 1.19.5    |
| requests                  | 2.25.1    |
| scikit-learn              | 0.24.2    |
| opencv-python             | 4.5.3.56  |
| tqdm                      | 4.62.3    |
| scipy                     | 1.5.4     |
| mxnet                     | latest    |

[requirements.txt](requirements.txt)


## Verify
```
usage: verification_onnx.py [-h] [-m ONNX_MODEL_PATH] [-b BIN_PATH]

Verify ONNX

optional arguments:
  -h, --help            show this help message and exit
  -m ONNX_MODEL_PATH, --onnx_model_path ONNX_MODEL_PATH
                        where is the onnx model.
  -b BIN_PATH, --bin_path BIN_PATH
                        where is the bin file.
```
