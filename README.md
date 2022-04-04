![SoyNet](https://user-images.githubusercontent.com/74886743/161455587-31dc85f4-d60c-4dd5-9612-113a9ac82c41.png)

# SoyNet Model Market

[SoyNet](https://soynet.io/en/) is an inference optimizing solution for AI models.

We start a [SoyNet AI model market](https://market.soymlops.com/#/about).

## SoyNet Overview

#### Core technology of SoyNet

- Accelerate model inference by maximizing the utilization of numerous cores on the GPU without compromising accuracy (2x to 5x compared to Tensorflow)
- Minimize GPU memory usage (1/5~1/15 level compared to Tensorflow)

   ※ Performance varies depends on the model and configuration environment.
   
#### Benefit of SoyNet

- can support customer to  provide AI applications and AI services in time (Time to Market)
- can help application developers to easily execute AI projects without additional technical AI knowledge and experience
- can help customer to reduce H/W (GPU, GPU server) or Cloud Instance cost for the same AI execution (inference)
- can support customer to respond to real-time environments that require very low latency in AI inference

#### Features of SoyNet

- Dedicated engine for inference of deep learning models
- Supports NVIDIA and non-NVIDIA GPUs (based on technologies such as CUDA and OpenCL, respectively)
- library files to be easiliy integrated with customer applications
dll file (Windows), so file (Linux) with header or *.lib for building in C/C++


## Folder Structure


```
   ├─data               : Example sample data
   ├─include            : File for using dll in Python
   ├─lib                : .dll or .so files for SoyNet
   ├─models             : SoyNet execution env
   │  └─model           : Model name
   │      ├─configs     : Model definitions (*.cfg)
   │      ├─engines     : SoyNet engine files
   │      ├─logs        : SoyNet log files
   │      └─weights     : Weight files for SoyNet models (*.weights)
   ├─samples            : Executable File
   └─utils              : Commonly-used functionand trial license
```
 - `engines` : it's made at the first time execution or when you modify the configs file.
 - `weights` : You can download .weights file from `download_soynet_weight.sh` in [weights folder](#folder-structure).
 - `license file` : Please contact [SoyNet](https://soynet.io/en/) if the time has passed.

## SoyNet Function.
 - `initSoyNet(.config, extend_param)` : Created a SoyNet handle.
 - `feedData(handle, data)` : Put the data into the SoyNet handle.
 - `inference(handle)` : Start inference.
 - `getOutput(handle, output)` : Put the inference data into the output.

   ※ `engine_serialize` in `extend_param`
      - This parameter determines whether to build a SoyNet engine or not.
      - If you run it for the first time, set engine_serialize to 1.
      - Also, if you edit the extend parameter, set engine_serialize to 1.
      - Set to 0 after engine is created.

## Requirements
#### H/W
 - GPU: RTX 3090 (NVIDA GPU with PASCAL architecture or higher)
 
   ※ You need to use .dll and .so files that match the GPU. Please contact [SoyNet](https://soynet.io/en/)

#### NVIDA Development Environment
 - CUDA (>= 11.1)
 - cuDNN (>= 8.x)
 - TensorRT (>= 8.x)

#### S/W
 - OS : Ubuntu 18.04 LTS
 - Others : OpenCV (for reading video files and outputting the screen)

## Getting Started
Before proceeding, please refer to the [Folder Structure](#folder-structure) to see if the required files exist.

1. You can download .weights file from `download_soynet_weight.sh` in [weights folder](#folder-structure).
2. Set [engine_serialize](#soynet-function) to 1 in the code of the samples file.
3. Just Run

## Model List
#### Classification
|Model|Link|
|---|---|
|[EfficientNet - Pytorch](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/EfficientNet_pytorch)||
|[EfficientNet - TensorFlow](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/EfficientNet_TensorFlow)||
|[Inception ResNet V2](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Inception_resnet_v2)||
|[VGG](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/VGG)||
|[SENet](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/SENet_legacy_senet)||
|[MobileNet V2](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Mobilenet_V2)||

#### Object Detection
|Model|Link|
|---|---|
|[Faster RCNN](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Detectron2_Faster-RCNN)||
|[RetinaFace](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/RetinaFace)||
|[EfficientDet](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/EfficientDet)||
|[SSD MobileNet](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/SSD_Mobilenet)||
|[Yolo V3](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Yolov3)||
|[Yolo V4](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Yolov4)||
|Yolo V5||
|Yolo V5 Face||
|Yolor||

#### Object Tracking
|Model|Link|
|---|---|
|[FairMot](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/FairMot)||

#### Pose Estimation
|Model|Link|
|---|---|
|[Pose RCNN](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Detectron2_Pose-RCNN)||
|[OpenPose](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Openpose-Darknet)||

#### Segmentation
|Model|Link|
|---|---|
|[Mask RCNN](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Detectron2_Mask-RCNN)||
|[Yolact](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Yolact)||
|[Yolact++](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Yolact%2B%2B)||

#### GAN
|Model|Link|
|---|---|
|[FAnoGan](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/FAnoGan)||
|[CycleGan](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/CycleGan)||
|[pix2pix](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/pix2pix)||
|[IDN](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/IDN)||
|[Glean](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/glean)||

#### NLP
|Model|Link|
|---|---|
|[Transformers MaskedLM](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Transformers_MaskedLM)||

#### ETC
|Model|Link|
|---|---|
|[ArcFace](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/ArcFace)||
|[Eca NFNet](https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Eca_NFNet)||



## Contact
For business inquiries or professional support requests please visit [SoyNet](https://market.soymlops.com/#/)
