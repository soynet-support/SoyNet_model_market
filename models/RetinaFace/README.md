# RetinaFace Model Overview
Face detection has advanced a lot, but face detection in natural conditions is still difficult. So we introduce RetinaFace, which is one-stage, and it has extras-supervised and self-supervised advantages to detect faces in pixels regardless of target size.

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
$ cd SoyNet_model_market/models/RetinaFace/weights && bash ./download_weights.sh
```

* Run
```
$ cd ../../../samples
$ python RetinaFace.py 
```

If you cannot create an engine, review the configuration settings again.

It is possible to create a C++ executable file.

Contact [SOYNET](https://market.soymlops.com/#/contact-us).

# Reference
 - [Original Code](https://github.com/biubug6/Pytorch_Retinaface)

# License for Original Model use

RetinaFace is under MIT License. 
See License terms and condition: [License](https://github.com/serengil/retinaface/blob/master/LICENSE)
