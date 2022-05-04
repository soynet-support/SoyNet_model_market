# Detectron2_Pose-RCNN Model Overview
Keypoint detection involves simultaneously detecting people and localizing their keypoints. Keypoints are the same thing as interest points. They are spatial locations, or points in the image that define what is interesting or what stand out in the image. They are invariant to image rotation, shrinkage, translation, distortion, and so on.

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
      - `class_count` : Number of classes.
      - `nms_count` : Number of NMS (Non-maximum Suppression).
      - `keypoint_count` : Number of Keypoint.

# Start SoyNet Demo Examples

* Clone github repository

```
$ git clone https://github.com/soynet-support/SoyNet_model_market.git
```

* download pre-trained weight files (already converted for SoyNet)

```
$ cd SoyNet_model_market/models/Detectron2_Pose-RCNN/weights && bash ./download_weights.sh
```

* Run
```
$ cd ../../../samples
$ python Detectron2_Pose-RCNN.py 
```

If you cannot create an engine, review the configuration settings again.

It is possible to create a C++ executable file.

Contact [SOYNET](https://market.soymlops.com/#/contact-us).

# Reference
 - [Original Code](https://github.com/facebookresearch/detectron2)


# Acknowlegement

Keypoint R-CNN is under Apache License. 
See License terms and condition: [License](https://github.com/facebookresearch/detectron2/blob/main/LICENSE)
