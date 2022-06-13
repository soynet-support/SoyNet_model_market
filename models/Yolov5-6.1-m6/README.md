# Yolov5m6 Model Overview
YOLOv5 is a family of object detection architectures and models pretrained on the COCO dataset, and represents Ultralytics open-source research into future vision AI methods, incorporating lessons learned and best practices evolved over thousands of hours of research and development.  
  
YOLOv5m6 is middle model in YOLOv5 and has P6 architectures.

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
      - `class_count` : Number of classes
      - `nms_count` : Number of NMS (Non-maximum Suppression)
      - `region_count` : Number to be used for NMS(Non-maximum Suppression)
      - `input_height`, `input_width` : Data size before entering preproc.
      - `model_size` : Data size before entering the model.
      - `cfg_file` : The path to cfg_file.
      - `weight_file` : The path to weight_file.
      - `engine_file` : The path to engine_file.
      - `log_file` :  The path to log_file.



# Start SoyNet Demo Examples

* Clone github repository

```
$ git clone https://github.com/soynet-support/SoyNet_model_market.git
```

* download pre-trained weight files (already converted for SoyNet)

```
$ cd SoyNet_model_market/models/Yolov5-6.1-m6/weights && bash ./download_weights.sh
```

* Run
```
$ cd ../../../samples
$ python Yolov5-6.1-m6.py 
```

If you cannot create an engine, review the configuration settings again.

It is possible to create a C++ executable file.

Contact [SOYNET](https://soynet.io/#/contact-us).

# Reference
 - [Original Code](https://github.com/ultralytics/yolov5)

# Acknowlegement

Yolov5 m6 is under GNU General Public License. 
See License terms and condition: [License](https://github.com/ultralytics/yolov5/blob/master/LICENSE)
