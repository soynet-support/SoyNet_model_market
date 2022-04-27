![SoyNet](https://user-images.githubusercontent.com/74886743/161455587-31dc85f4-d60c-4dd5-9612-113a9ac82c41.png)

# SoyNet Model Market

[SoyNet](https://soynet.io/en/) is an inference optimizing solution for AI models.

We start a [SoyNet AI model market](https://market.soymlops.com/#/about).

## SoyNet Overview

#### Core technology of SoyNet

- Accelerate model inference by maximizing the utilization of numerous cores on the GPU without compromising accuracy (2x to 5x compared to Tensorflow).
- Minimize GPU memory usage (1/5~1/15 level compared to Tensorflow).

   ※ Performance varies depends on the model and configuration environment.
   
#### Benefit of SoyNet

- can support customer to  provide AI applications and AI services in time. (Time to Market)
- can help application developers to easily execute AI projects without additional technical AI knowledge and experience.
- can help customer to reduce H/W (GPU, GPU server) or Cloud Instance cost for the same AI execution. (inference)
- can support customer to respond to real-time environments that require very low latency in AI inference.

#### Features of SoyNet

- Dedicated engine for inference of deep learning models.
- Supports NVIDIA and non-NVIDIA GPUs (based on technologies such as CUDA and OpenCL, respectively).
- library files to be easiliy integrated with customer applications
dll file (Windows), so file (Linux) with header or *.lib for building in C/C++.
- We can provide c++ and python executable files.


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
 - `weights` : You can download .weights file from `download_soynet_weight.py` in [weights folder](#folder-structure).
 - `license file` : Please contact [SoyNet](https://soynet.io/en/) if the time has passed.

## SoyNet Function.
 - `initSoyNet(.cfg, extend_param)` : Created a SoyNet handle.
 - `feedData(handle, data)` : Put the data into the SoyNet handle.
 - `inference(handle)` : Start inference.
 - `getOutput(handle, output)` : Put the inference data into the output.


   ※ `extend_param`
      - `extend_param` contains parameters necessary to define the model, such as input size, engine_serialize, batch_size ...
      - The parameters required may vary depending on the model.

   ※ `engine_serialize` in `extend_param`
      - This parameter determines whether to build a SoyNet engine or not.
      - If you run it for the first time, set engine_serialize to 1.
      - Also, if you edit the extend parameter, set engine_serialize to 1.
      - Set to 0 after engine is created.

## Prerequisites
#### NVIDA Development Environment
 - CUDA (= 11.1)
 - cuDNN (>= 8.x)
 - TensorRT (= 8.2.1.8)
 
    ※ You need to use .dll and .so files that match CDUA and TensorRT versions. If you want another version, Please contact [SoyNet](https://soynet.io/en/).

#### S/W
 - OS : Ubuntu 18.04 LTS
 - Others : 
   - OpenCV (for reading video files and outputting the screen)
   - Wget (for 'download_soynet_weight.sh' in [weights folder](#folder-structure)

## Getting Started
Before proceeding, please refer to the [Folder Structure](#folder-structure) to see if the required files exist.

1. You can download .weights file from `download_soynet_weight.py` in [weights folder](#folder-structure).
2. Set [engine_serialize](#soynet-function) to 1 in the code of the samples file.
3. Just Run

The models folder contains a detailed description.

## Model List
#### Classification

<table>
  <tr>
    <th rowspan="2" align=center width="300">Model</td>
    <th rowspan="2" align=center width="200">Site Reference</td>
    <th colspan="3" align=center width="600">Support Platform</td>
      <tr>
         <th align=center width="200">X86_Linux</td>
         <th align=center width="200">X86_Windows</td>
         <th align=center width="200">Jetson_Linux</td>
      </tr>
  </tr>
    <tr>
    <td align=center><a href="https://github.com/soynet-support/SoyNet_model_market/tree/main/models/EfficientNet_TensorFlow">EfficientNet - TensorFlow</a></td>
    <td align=center><a href="https://github.com/qubvel/efficientnet#models">LINK</a></td>
   <td align=center>✔</td>
   <td align=center>✔</td>
   <td align=center></td>
  </tr>
  </tr>
    <tr>
    <td align=center><a href="https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Inception_resnet_v2">Inception ResNet V2</a></td>
    <td align=center></td>
    <td align=center>✔</td>
    <td align=center>✔</td>
    <td align=center></td>
  </tr>
  <tr>
    <td align=center><a href="https://github.com/soynet-support/SoyNet_model_market/tree/main/models/VGG">VGG</a></td>
    <td align=center></td>
    <td align=center>✔</td>
    <td align=center>✔</td>
    <td align=center></td>
  </tr>
   <tr>
    <td align=center><a href="https://github.com/soynet-support/SoyNet_model_market/tree/main/models/SENet_legacy_senet">SENet</a></td>
    <td align=center></td>
    <td align=center>✔</td>
    <td align=center>✔</td>
    <td align=center></td>
  </tr>
  <tr>
    <td align=center><a href="https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Mobilenet_V2">MobileNet V2</a></td>
    <td align=center><a href="https://pytorch.org/vision/stable/_modules/torchvision/models/mobilenetv2.html">LINK</a></td>
    <td align=center>✔</td>
    <td align=center>✔</td>
    <td align=center></td>
  </tr>
</table>

#### Object Detection
<table>
  <tr>
    <th rowspan="2" align=center width="300">Model</td>
    <th rowspan="2" align=center width="200">Site Reference</td>
    <th colspan="3" align=center width="600">Support Platform</td>
      <tr>
         <th align=center width="200">X86_Linux</td>
         <th align=center width="200">X86_Windows</td>
         <th align=center width="200">Jetson_Linux</td>
      </tr>
  </tr>
  <tr>
    <td align=center><a href="https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Detectron2_Faster-RCNN">Faster RCNN</a></td>
    <td align=center><a href="https://github.com/facebookresearch/detectron2">LINK</a></td>
    <td align=center>✔</td>
    <td align=center>✔</td>
    <td align=center></td>
  </tr>
  <tr>
    <td align=center><a href="https://github.com/soynet-support/SoyNet_model_market/tree/main/models/RetinaFace">RetinaFace</a></td>
    <td align=center><a href="https://github.com/biubug6/Pytorch_Retinaface">LINK</a></td>
    <td align=center>✔</td>
    <td align=center>✔</td>
    <td align=center></td>
  </tr>
  <tr>
    <td align=center><a href="https://github.com/soynet-support/SoyNet_model_market/tree/main/models/EfficientDet">EfficientDet</a></td>
    <td align=center><a href="https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch">LINK</a></td>
    <td align=center>✔</td>
    <td align=center>✔</td>
    <td align=center></td>
  </tr>
  <tr>
    <td align=center><a href="https://github.com/soynet-support/SoyNet_model_market/tree/main/models/SSD_Mobilenet">SSD MobileNet</a></td>
    <td align=center><a href="https://github.com/tensorflow/models/tree/master/research/object_detection">LINK</a></td>
    <td align=center>✔</td>
    <td align=center>✔</td>
    <td align=center></td>
  </tr>
  <tr>
    <td align=center><a href="https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Yolov3">Yolo V3</a></td>
    <td align=center><a href="https://github.com/AlexeyAB/darknet">LINK</a></td>
    <td align=center>✔</td>
    <td align=center>✔</td>
    <td align=center></td>
  </tr>
  <tr>
    <td align=center><a href="https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Yolov4">Yolo V4</a></td>
    <td align=center><a href="https://github.com/AlexeyAB/darknet">LINK</a></td>
    <td align=center>✔</td>
    <td align=center>✔</td>
    <td align=center></td>
  </tr>
  <tr>
    <td align=center><a href="https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Yolov5-6.1-l">Yolo V5-l</a></td>
    <td align=center><a href="https://github.com/ultralytics/yolov5/releases/tag/v6.1">LINK</a></td>
    <td align=center>✔</td>
    <td align=center>✔</td>
    <td align=center></td>
  </tr>
  <tr>
    <td align=center><a href="https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Yolov5-6.1-m">Yolo V5-m</a></td>
    <td align=center><a href="https://github.com/ultralytics/yolov5/releases/tag/v6.1">LINK</a></td>
    <td align=center>✔</td>
    <td align=center>✔</td>
    <td align=center></td>
  </tr>
  <tr>
    <td align=center><a href="https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Yolov5-6.1-n">Yolo V5-n</a></td>
    <td align=center><a href="https://github.com/ultralytics/yolov5/releases/tag/v6.1">LINK</a></td>
    <td align=center>✔</td>
    <td align=center>✔</td>
    <td align=center></td>
  </tr>
  <tr>
    <td align=center><a href="https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Yolov5-6.1-s">Yolo V5-s</a></td>
    <td align=center><a href="https://github.com/ultralytics/yolov5/releases/tag/v6.1">LINK</a></td>
    <td align=center>✔</td>
    <td align=center>✔</td>
    <td align=center></td>
  </tr>
  <tr>
    <td align=center><a href="https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Yolov5-6.1-x">Yolo V5-x</a></td>
    <td align=center><a href="https://github.com/ultralytics/yolov5/releases/tag/v6.1">LINK</a></td>
    <td align=center>✔</td>
    <td align=center>✔</td>
    <td align=center></td>
  </tr>
  <tr>
    <td align=center><a href="https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Yolov5-6.1-l6">Yolo V5-l6</a></td>
    <td align=center><a href="https://github.com/ultralytics/yolov5/releases/tag/v6.1">LINK</a></td>
    <td align=center>✔</td>
    <td align=center>✔</td>
    <td align=center></td>
  </tr>
  <tr>
    <td align=center><a href="https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Yolov5-6.1-m6">Yolo V5-m6</a></td>
    <td align=center><a href="https://github.com/ultralytics/yolov5/releases/tag/v6.1">LINK</a></td>
    <td align=center>✔</td>
    <td align=center>✔</td>
    <td align=center></td>
  </tr>
  <tr>
    <td align=center><a href="https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Yolov5_Face">Yolo V5 Face</a></td>
    <td align=center><a href="https://morioh.com/p/b101507afdf5">LINK</a></td>
    <td align=center>✔</td>
    <td align=center>✔</td>
    <td align=center></td>
  </tr>
  <tr>
    <td align=center><a href="https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Yolor">Yolor</a></td>
    <td align=center><a href="https://github.com/WongKinYiu/yolor">LINK</a></td>
    <td align=center>✔</td>
    <td align=center>✔</td>
    <td align=center></td>
  </tr>
</table>

#### Object Tracking
<table>
  <tr>
    <th rowspan="2" align=center width="300">Model</td>
    <th rowspan="2" align=center width="200">Site Reference</td>
    <th colspan="3" align=center width="600">Support Platform</td>
      <tr>
         <th align=center width="200">X86_Linux</td>
         <th align=center width="200">X86_Windows</td>
         <th align=center width="200">Jetson_Linux</td>
      </tr>
  </tr>
  <tr>
    <td align=center><a href="https://github.com/soynet-support/SoyNet_model_market/tree/main/models/FairMot">FairMot</a></td>
    <td align=center><a href="https://github.com/ifzhang/FairMOT">LINK</a></td>
    <td align=center>✔</td>
    <td align=center>✔</td>
    <td align=center></td>
  </tr>
</table>

#### Pose Estimation
<table>
  <tr>
    <th rowspan="2" align=center width="300">Model</td>
    <th rowspan="2" align=center width="200">Site Reference</td>
    <th colspan="3" align=center width="600">Support Platform</td>
      <tr>
         <th align=center width="200">X86_Linux</td>
         <th align=center width="200">X86_Windows</td>
         <th align=center width="200">Jetson_Linux</td>
      </tr>
  </tr>
  <tr>
    <td align=center><a href="https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Detectron2_Pose-RCNN">Pose RCNN</a></td>
    <td align=center><a href="https://github.com/facebookresearch/detectron2">LINK</a></td>
    <td align=center>✔</td>
    <td align=center>✔</td>
    <td align=center></td>
  </tr>
  <tr>
    <td align=center><a href="https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Openpose-Darknet">OpenPose</a></td>
    <td align=center><a href="https://github.com/lincolnhard/openpose-darknet">LINK</a></td>
    <td align=center>✔</td>
    <td align=center>✔</td>
    <td align=center></td>
  </tr>
</table>

#### Segmentation
<table>
  <tr>
    <th rowspan="2" align=center width="300">Model</td>
    <th rowspan="2" align=center width="200">Site Reference</td>
    <th colspan="3" align=center width="600">Support Platform</td>
      <tr>
         <th align=center width="200">X86_Linux</td>
         <th align=center width="200">X86_Windows</td>
         <th align=center width="200">Jetson_Linux</td>
      </tr>
  </tr>
  <tr>
    <td align=center><a href="https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Detectron2_Mask-RCNN">Mask RCNN</a></td>
    <td align=center><a href="https://github.com/facebookresearch/detectron2">LINK</a></td>
    <td align=center>✔</td>
    <td align=center>✔</td>
    <td align=center></td>
  </tr>
  <tr>
    <td align=center><a href="https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Yolact">Yolact</a></td>
    <td align=center><a href="https://github.com/dbolya/yolact">LINK</a></td>
    <td align=center>✔</td>
    <td align=center>✔</td>
    <td align=center></td>
  </tr>
  <tr>
    <td align=center><a href="https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Yolact%2B%2B">Yolact++</a></td>
    <td align=center><a href="https://github.com/dbolya/yolact">LINK</a></td>
    <td align=center>✔</td>
    <td align=center>✔</td>
    <td align=center></td>
  </tr>
</table>

#### GAN
<table>
  <tr>
    <th rowspan="2" align=center width="300">Model</td>
    <th rowspan="2" align=center width="200">Site Reference</td>
    <th colspan="3" align=center width="600">Support Platform</td>
      <tr>
         <th align=center width="200">X86_Linux</td>
         <th align=center width="200">X86_Windows</td>
         <th align=center width="200">Jetson_Linux</td>
      </tr>
  </tr>
  <tr>
    <td align=center><a href="https://github.com/soynet-support/SoyNet_model_market/tree/main/models/FAnoGan">FAnoGan</a></td>
    <td align=center><a href="https://github.com/mulkong/f-AnoGAN_with_Pytorch">LINK</a></td>
    <td align=center>✔</td>
    <td align=center>✔</td>
    <td align=center></td>
  </tr>
  <tr>
    <td align=center><a href="https://github.com/soynet-support/SoyNet_model_market/tree/main/models/CycleGan">CycleGan</a></td>
    <td align=center><a href="https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix">LINK</a></td>
    <td align=center>✔</td>
    <td align=center>✔</td>
    <td align=center></td>
  </tr>
  <tr>
    <td align=center><a href="https://github.com/soynet-support/SoyNet_model_market/tree/main/models/pix2pix">Pix2Pix</a></td>
    <td align=center><a href="https://github.com/phillipi/pix2pix">LINK</a></td>
    <td align=center>✔</td>
    <td align=center>✔</td>
    <td align=center></td>
  </tr>
  <tr>
    <td align=center><a href="https://github.com/soynet-support/SoyNet_model_market/tree/main/models/IDN">IDN</a></td>
    <td align=center><a href="https://github.com/yjn870/IDN-pytorch">LINK</a></td>
    <td align=center>✔</td>
    <td align=center>✔</td>
    <td align=center></td>
  </tr>
  <tr>
    <td align=center><a href="https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Glean">Glean</a></td>
    <td align=center><a href="https://github.com/open-mmlab/mmediting/tree/master/configs/restorers/glean">LINK</a></td>
    <td align=center>✔</td>
    <td align=center>✔</td>
    <td align=center></td>
  </tr>
</table>

#### NLP
<table>
  <tr>
    <th rowspan="2" align=center width="300">Model</td>
    <th rowspan="2" align=center width="200">Site Reference</td>
    <th colspan="3" align=center width="600">Support Platform</td>
      <tr>
         <th align=center width="200">X86_Linux</td>
         <th align=center width="200">X86_Windows</td>
         <th align=center width="200">Jetson_Linux</td>
      </tr>
  </tr>
  <tr>
    <td align=center><a href="https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Transformers_MaskedLM">Transformers MaskedLM</a></td>
    <td align=center><a href="https://github.com/huggingface/transformers">LINK</a></td>
    <td align=center>✔</td>
    <td align=center>✔</td>
    <td align=center></td>
  </tr>
</table>

#### ETC
<table>
  <tr>
    <th rowspan="2" align=center width="300">Model</td>
    <th rowspan="2" align=center width="200">Site Reference</td>
    <th colspan="3" align=center width="600">Support Platform</td>
      <tr>
         <th align=center width="200">X86_Linux</td>
         <th align=center width="200">X86_Windows</td>
         <th align=center width="200">Jetson_Linux</td>
      </tr>
  </tr>
  <tr>
    <td align=center><a href="https://github.com/soynet-support/SoyNet_model_market/tree/main/models/ArcFace">ArcFace</a></td>
    <td align=center><a href="https://github.com/ronghuaiyang/arcface-pytorch">LINK</a></td>
    <td align=center>✔</td>
    <td align=center>✔</td>
    <td align=center></td>
  </tr>
  <tr>
    <td align=center><a href="https://github.com/soynet-support/SoyNet_model_market/tree/main/models/Eca_NFNet">Eca NFNet</a></td>
    <td align=center><a href="https://www.kaggle.com/h053473666/2class-infer-batch-20">LINK</a></td>
    <td align=center>✔</td>
    <td align=center>✔</td>
    <td align=center></td>
  </tr>
</table>

The model will be added continuously, so please contact [SoyNet](https://soynet.io/en/) for the **Custom Model**.

## Contact
For business inquiries or professional support requests please visit [SoyNet](https://market.soymlops.com/#/).

