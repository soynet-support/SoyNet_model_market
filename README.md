![SoyNet](https://user-images.githubusercontent.com/74886743/161455587-31dc85f4-d60c-4dd5-9612-113a9ac82c41.png)

# SoyNet Model Market

[SoyNet](https://soynet.io/) is an inference optimizing solution for AI models.

This section describes the process of performing a demo running Yolo (v3-tiny, v3, v4), one of most famous object detection models.

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
 - engines: it's made at the first time execution or when you modify the configs file.
 - weights: you can download [SoyNet](https://soynet.io/)
 - license file: Please contact [SoyNet](https://soynet.io/) if the time has passed.

## Requirements
#### H/W
 - GPU: RTX 3090 (NVIDA GPU with PASCAL architecture or higher)
 
   ※ You need to use dll and so files that match the GPU. Pleas Contact [SoyNet](https://soynet.io/)

#### NVIDA Development Environment
 - CUDA (>= 11.1)
 - cuDNN (>= 8.x)
 - TensorRT (>= 8.x)

#### S/W
 - OS : Ubuntu 18.04 LTS
 - Others : OpenCV (for reading video files and outputting the screen)

## Getting Started
Before proceeding, please refer to the [Folder Structure](#folder-structure) to see if the required files exist.
1. You can download .weights file from [SoyNet](https://soynet.io/). Put it in the model folder you want to use.
2. Set engine_serialize to 1 in the code of the samples file.
3. Just Run

## Model List
#### Classification
|Model|Link|
|---|---|
|EfficientNet - Pytorch||
|EfficientNet - TensorFlow||
|Inception ResNet V2||
|VGG16||
|SENet||
|MobileNet V2||

#### Object Detection
|Model|Link|
|---|---|
|Faster RCNN||
|RetinaFace||
|EfficientDet||
|SSD MobileNet||
|Yolo V3||
|Yolo V4||
|Yolo V5||
|Yolo V5 Face||
|Yolor||

#### Object Tracking
|Model|Link|
|---|---|
|FairMot||

#### Pose Estimation
|Model|Link|
|---|---|
|Pose RCNN||
|OpenPose||

#### Segmentation
|Model|Link|
|---|---|
|Mask RCNN||
|Yolact||
|Yolact++||

#### GAN
|Model|Link|
|---|---|
|FAnoGan||
|CycleGan||
|pix2pix||
|IDN||
|Glean||

#### NLP
|Model|Link|
|---|---|
|Transformers MaskedLM||

#### ETC
|Model|Link|
|---|---|
|ArcFace||
|Eca NFNet||



## Contact
For business inquiries or professional support requests please visit [SoyNet](https://market.soymlops.com/#/)
