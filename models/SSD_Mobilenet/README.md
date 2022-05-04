# SSD_Mobilenet Model Overview
SSD Mobilenet V2 is a one-stage object detection model which has gained popularity for its lean network and novel depthwise separable convolutions. It is a model commonly deployed on low compute devices such as mobile (hence the name Mobilenet) with high accuracy performance.

# Prerequisites

#### NVIDA Development Environment
 - CUDA (= 11.1)
 - cuDNN (>= 8.x)
 - TensorRT (= 8.2.1.8)
 
    ※ You need to use .dll and .so files that match CDUA and TensorRT versions. If you want another version, Please contact [SoyNet](https://soynet.io/en/).
#### S/W
 - OS : Ubuntu 18.04 LTS
 - Others : OpenCV (for reading video files and outputting the screen)


# Parameters
 - `extend_param`
      - `batch_size` : This is the batch-size of the data you want to input.
      - `engine_serialize` : Whether or not the engine is created. (default : 0)
         - 0: Load existing engine file.
         - 1 : Create engine file from weight file. you need to set value to in following cases.
            - Change extended param.
            - Change weight file.
      - `cfg_file` : The path to cfg_file.
      - `weight_file` : The path to weight_file.
      - `engine_file` : The path to engine_file.
      - `log_file` :  The path to log_file.
      - `model_height`, `model_width` : Data size before entering the model.
      - `class_count` : Number of classes.
      - `nms_count` : Number of NMS (Non-maximum Suppression).
      - `region_count` : Number of region proposal.

# Start SoyNet Demo Examples

* Clone github repository

```
$ git clone https://github.com/soynet-support/SoyNet_model_market.git
```

* download pre-trained weight files (already converted for SoyNet)

```
$ cd SoyNet_model_market/models/SSD_Mobilenet/weights && bash ./download_weights.sh
```

* Run
```
$ cd ../../../samples
$ python SSD_Mobilenet.py 
```

If you cannot create an engine, review the configuration settings again.

It is possible to create a C++ executable file.

Contact [SOYNET](https://market.soymlops.com/#/contact-us).

# Reference
 - [Original Code](https://github.com/tensorflow/models/tree/master/research/object_detection)

# Acknowlegement

SSD Mobilenet V2 is under MIT License. 
See License terms and condition: [License](https://github.com/chuanqi305/MobileNet-SSD/blob/master/LICENSE)

