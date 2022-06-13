# Faster RCNN Model Overview
Because Fast R-CNN models still extract region proposals through the Selective search algorithm, there is a limit to improving learning and detection speed.
To address these issues, Faster R-CNN introduces the Region Proposal Network (RPN), a network that performs candidate region extraction operations.

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
      - `weight_file` : The path to weight_file.
      - `engine_file` : The path to engine_file.
      - `log_file` :  The path to log_file.
      - `input_height`, `input_width` : Data size before entering preproc.
      - `class_count` : Number of classes.
      - `nms_count` : Number of NMS (Non-maximum Suppression).

# Start SoyNet Demo Examples

* Clone github repository

```
$ git clone https://github.com/soynet-support/SoyNet_model_market.git
```

* download pre-trained weight files (already converted for SoyNet)

```
$ cd SoyNet_model_market/models/Detectron2_Faster-RCNN/weights && bash ./download_weights.sh
```

* Run
```
$ cd ../../../samples
$ python Detectron2_Faster-RCNN.py 
```

If you cannot create an engine, review the configuration settings again.

It is possible to create a C++ executable file.

Contact [SOYNET](https://soynet.io/#/contact-us).

# Reference
 - [Original Code](https://github.com/facebookresearch/detectron2)
 - [Paper](https://arxiv.org/abs/1506.01497)


# Acknowlegement

Faster R-CNN is under Apache License. 
See License terms and condition: [License](https://github.com/facebookresearch/detectron2/blob/main/LICENSE)
