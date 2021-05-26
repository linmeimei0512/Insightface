
# Insightface

## Recent Update

**`2021-05-25`**: Add mask renderer. [here](./tools/mask_renderer)

**`2021-05-25`**: Add face recognition. [here](./recognition/arcface_torch)

**`2021-05-24`**: Add model-zoo. [here](./model_zoo)

**`2021-05-24`**: Add face detection - MTCNN. [here](./detection)

**`2021-05-20`**: Add pytorch covert to tensorflow lite. [here](./tools/model_converter)

**`2021-05-19`**: Fix generate pairs.txt with generate_pairs.py. Use 10-fold cross-validation.

**`2021-05-07`**: First commit.



## Content

[Face Detection](./detection)
- [MTCNN](./detection/mtcnn)

[Face Recognition](./recognition)
- [Arcface-PyTorch](./recognition/arcface_torch)

[Generate Datasets](./tools/generate_mxnet_dataset)
- [Generate MXNet Datasets](./tools/generate_mxnet_dataset)

[Model Converter](./tools/model_converter)
- [PyTorch to Tensorflow Lite](./tools/model_converter/)
- [PyTorch to ONNX](./tools/model_converter/pytorch_to_onnx_converter.py)
- [ONNX to Tensorflow pb](./tools/model_converter/onnx_to_pb_converter.py)
- [Tensorflow pb to Tensorflow Lite](./tools/model_converter/pb_to_tflite_converter.py)

[Mask Renderer](./tools/mask_renderer)