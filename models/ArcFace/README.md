# ArcFace Model Overview
Arc Face is face recognition Model.
This proposed a new loss function to improve facial recognition performance.
One of the important challenges of Feature learning using deep CNNs in Face Recognition is designing appropriate loss functions that improve discernment.

# Prerequisites
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


# Parameters
 - `extend_param`
      - `batch_size` : This is the batch-size of the data you want to input.
      - `engine_serialize` : Whether or not the engine is created.
         - This parameter determines whether to build a SoyNet engine or not.
         - If you run it for the first time, set engine_serialize to 1.
         - Also, if you edit the extend parameter, set engine_serialize to 1.
         - Set to 0 after engine is created.
      - `cfg_file` : The path to cfg_file.
      - `weight_file` : The path to weight_file.
      - `engine_file` : The path to engine_file.
      - `log_file` :  The path to log_file.
      - `model_height`, `model_width` : Data size before entering the model.

# Install SoyNet Demo

* Clone github repository

```
$ git clone https://github.com/soynet-support/SoyNet_model_market.git
```

* download pre-trained weight files (already converted to SoyNet)

```
$ cd ~/soynetmodelzoo/yolov5/mgmt/weights && bash ./download_weights.sh
```

* set environment parameter

```
$ LD_LIBRARY_PATH=~/SoyNet_model_market/lib:$LD_LIBRARY_PATH
```

* Run
```
$ python ArcFace.py 
```

It is possible to create a C++ executable file.
Contact [SoyNet](https://soynet.io/en/)


# Reference
 - [Original Code](https://github.com/ronghuaiyang/arcface-pytorch)
 - [Paper](https://arxiv.org/abs/1801.07698)
