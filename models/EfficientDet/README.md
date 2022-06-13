# EfficientDet Model Overview
EfficientDet is a type of object detection model, which utilizes several optimization and backbone tweaks, such as the use of a BiFPN, and a compound scaling method that uniformly scales the resolution,depth and width for all backbones, feature networks and box/class prediction networks at the same time.

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
      - `input_height`, `input_width` : Data size before entering preproc.
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
$ cd SoyNet_model_market/models/EfficientDet/weights && bash ./download_weights.sh
```

* Run
```
$ cd ../../../samples
$ python EfficientDet.py 
```

If you cannot create an engine, review the configuration settings again.

It is possible to create a C++ executable file.

Contact [SOYNET](https://soynet.io/#/contact-us).

# Reference
 - [Original Code](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)


# Acknowlegement

EfficientDet is under GNU Lesser General Public License. 
See License terms and condition: [License](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/blob/master/LICENSE)
