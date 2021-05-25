
# Model Zoo

## Face Detection

### 1. MTCNN
Model: [Dropbox](https://www.dropbox.com/s/m70u2irvtihxpr8/mtcnn.pb?dl=0)



## Face Recognition

### 1. Arcface

#### Glint360k - PyTorch 
[Dropbox](https://www.dropbox.com/sh/vayzcw1d13m3no0/AADS9ivFpij1eoFQ1BjhGCNZa?dl=0)

|   Datasets        | log                                                                   |Model                                                                          |backbone       | IJBC(1e-05)   | IJBC(1e-04)   |agedb30    |cfp_fp     |lfw        | 
| :---:             | :---                                                                  |:---                                                                           |:---           | :---          | :---          |:---       |:---       |:---       |
| Glint360k-Cosface |[log](https://www.dropbox.com/s/1cx3jjo3csw2ls6/training.log?dl=0)     | [Dropbox](https://www.dropbox.com/s/i7ubb4ho4e0a4mn/weight_backbone.pth?dl=0) |r18-fp16-0.1   | 93.16         | 95.33         | 97.72     | 97.73     | 99.77     |
| Glint360k-Cosface |[log](https://www.dropbox.com/s/k2z32anwr51rm7j/training.log?dl=0)     | [Dropbox](https://www.dropbox.com/s/sf3fn8niuk14pbj/weight_backbone.pth?dl=0) |r34-fp16-0.1   | 95.16         | 96.56         | 98.33     | 98.78     | 99.82     |
| Glint360k-Cosface |[log](https://www.dropbox.com/s/l8ct5iwpbw9hscv/training.log?dl=0)     | [Dropbox](https://www.dropbox.com/s/ya45pl9svmkn0le/weight_backbone.pth?dl=0) |r50-fp16-0.1   | 95.61         | 96.97         | 98.38     | 99.20     | 99.83     |
| Glint360k-Cosface |[log](https://www.dropbox.com/s/d47nisxiorv1090/training.log?dl=0)     | [Dropbox](https://www.dropbox.com/s/rl7c0vwgksvv7oi/weight_backbone.pth?dl=0) |r100-fp16-0.1  | 95.88         | 97.32         | 98.48     | 99.29     | 99.82     |
  

#### MS1MV3 - PyTorch 
[Dropbox](https://www.dropbox.com/sh/vayzcw1d13m3no0/AADS9ivFpij1eoFQ1BjhGCNZa?dl=0)

|   Datasets        | log                                                                   |Model                                                                          |backbone       | IJBC(1e-05)   | IJBC(1e-04)   |agedb30    |cfp_fp     |lfw        | 
| :---:             | :---                                                                  |:---                                                                           |:---           | :---          | :---          |:---       |:---       |:---       |
| MS1MV3-Arcface    |[log](https://www.dropbox.com/s/fizspv92zcndor3/training.log?dl=0)     | [Dropbox](https://www.dropbox.com/s/iktpu9jl5g4g84g/weight_backbone.pth?dl=0) | r18-fp16      | 92.07         | 94.66         | 97.77     | 97.73     | 99.77     |
| MS1MV3-Arcface    |[log](https://www.dropbox.com/s/k2z32anwr51rm7j/training.log?dl=0)     | [Dropbox](https://www.dropbox.com/s/sf3fn8niuk14pbj/weight_backbone.pth?dl=0) | r34-fp16      | 94.10         | 95.90         | 98.10     | 98.67     | 99.80     |
| MS1MV3-Arcface    |[log](https://www.dropbox.com/s/7lve18sei96bunq/training.log?dl=0)     | [Dropbox](https://www.dropbox.com/s/z80q91uxhxzeodz/weight_backbone.pth?dl=0) | r50-fp16      | 94.79         | 96.46         | 98.35     | 98.96     | 99.83     | 
| MS1MV3-Arcface    |[log](https://www.dropbox.com/s/4g1dbdhkjaw3qt4/training.log?dl=0)     | [Dropbox](https://www.dropbox.com/s/ogt5u4j6raa45tj/weight_backbone.pth?dl=0) | r100-fp16     | 95.31         | 96.81         | 98.48     | 99.06     | 99.85     | 

