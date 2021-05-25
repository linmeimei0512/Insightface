
# Face Recognition

## Recent Update

**`2021-05-25`**: First commit.



## Content
[Face Recognition]()
- [Set up environment](#set-up)
- [Download model](#download-model)
- [Face recognition]()



## Set Up
[Set up environment](#set-up)
- [Create python environment](#create-python-environment)
- [Install PyTorch](#install-pytorch)
- [Install python package](#install-python-package)



## Create Python Environment

| Python                    | 3.6       | 
| :---                      | :---      |

```
conda create -n insightface python=3.6
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
| torchsummary              | 1.5.1     |
| mxnet                     | latest    |

[requirements.txt](requirements.txt)
```
pip install -r requirements.txt
```

#### GPU Version (CUDA 10.1)

| Package                   | Version   | 
| :---                      | :---      |
| easydict                  | 1.9       |
| numpy                     | 1.19.5    |
| requests                  | 2.25.1    |
| scikit-learn              | 0.24.2    |
| torchsummary              | 1.5.1     |
| mxnet-cu101               | 1.5.0     |

[requirements-gpu.txt](requirements-gpu.txt)
```
pip install -r requirements-gpu.txt
```



## Download Model
[Model-Zoo](../../model_zoo/README.md)

Copy model to ./model_zoo/Arcface/PyTorch folder.



## Face Recognition

```
usage: face_recognition.py [-h] [--network NETWORK] [--model_path MODEL_PATH]
                           [--weight_path WEIGHT_PATH]
                           [--image_1_path IMAGE_1_PATH]
                           [--image_2_path IMAGE_2_PATH]

Face recognition

optional arguments:
  -h, --help            show this help message and exit
  --network NETWORK     
                        backbone network. default: r100
  --model_path MODEL_PATH
                        where is the insightface torch model. default: None
  --weight_path WEIGHT_PATH
                        where is the insightface torch weight path.
  --image_1_path IMAGE_1_PATH
                        where is the image 1 path.
  --image_2_path IMAGE_2_PATH
                        where is the image 2 path.
```

```
python face_recognition.py --network r100 --weight_path ../../model_zoo/Arcface/PyTorch/glint360k_cosface_r100_fp16_0.1/weight_backbone.pth --image_1_path ../../images/Tom_Hanks_01.jpg --image_2_path ../../images/Tom_Hanks_02.jpg
```



## Training
[Training](#training)
- [Download datasets](#1-download-datasets)
- [Edit config](#1-edit-config)
- [Train](#2-train)
- [Pretrain](#3-pretrain)

### 1. Download datasets
[Dataset-Zoo](https://github.com/deepinsight/insightface/wiki/Dataset-Zoo)

Download the datasets and place it in *`$INSIGHTFACE_ROOT/datasets/`*. Each training dataset includes at least following 6 files:

```
    faces_emore/
       train.idx
       train.rec
       property
       lfw.bin
       cfp_fp.bin
       agedb_30.bin
```


### 2. Edit config
```

config.dataset = "webface"                  # use dataset
config.embedding_size = 512                 # embedding size
config.sample_rate = 1                      # sample rate
config.fp16 = False                         # use fp16
config.momentum = 0.9                       # momentum
config.weight_decay = 5e-4                  # weight decay
config.batch_size = 64                      # batch size
config.lr = 0.1                             # learn rate
config.output = "ms1mv3_arcface_r50"        # output path

```

### 3. Train
```
usage: train.py [-h] [--local_rank LOCAL_RANK] [--network NETWORK]
                [--loss LOSS] [--resume RESUME]

PyTorch ArcFace Training

optional arguments:
  -h, --help            show this help message and exit
  --local_rank LOCAL_RANK
                        local_rank
  --network NETWORK     backbone network (iresnet18, iresnet34, iresnet50, iresnet100, iresnet200)
  --loss LOSS           loss function (Cosface, Arcface)
  --resume RESUME       model resuming

```

#### Single node, 1 GPUs
```
python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train.py
```

#### Single node, 4 GPUs
```
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train.py
```

#### Multiple nodes, each node 8 GPUs
Node 0:
```
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr="ip1" --master_port=1234 train.py
```

Node 1:
```
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr="ip1" --master_port=1234 train.py
```


### 4. Pretrain
Copy pytorch weight (weight_backbone.pth) to output folder.

```
python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1234 train.py --resume 1
```