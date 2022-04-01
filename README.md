# SoyNet_model_market

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
   │      └─weights     : Weight files for SoyNet models
   ├─samples            : Folder to build sample demo 
   └─utils              : Commonly-used functionand trial license
```
__- engines:__ it's made at the first time execution or when you modify the configs file.

## Contact
For business inquiries or professional support requests please visit https://market.soymlops.com/#/
