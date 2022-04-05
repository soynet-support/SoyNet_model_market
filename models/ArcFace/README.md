# ArcFace Model Overview
Arc Face is face recognition Model.
This proposed a new loss function to improve facial recognition performance.
One of the important challenges of Feature learning using deep CNNs in Face Recognition is designing appropriate loss functions that improve discernment.

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
$ git clone https://github.com/soynetmodelzoo/yolov5 ~/soynetmodelzoo/yolov5/
```

* download pre-trained weight files (already converted to SoyNet)

```
$ cd ~/soynetmodelzoo/yolov5/mgmt/weights && bash ./download_weights.sh
```

* set environment parameter

```
$ LD_LIBRARY_PATH=~/soynetmodelzoo/yolov5/mgmt:$LD_LIBRARY_PATH
```

* Demo code Build and run yolo demo (for C++ only)

```
$ cd /demo_yolo/samples && make all
$ ./yolov5            
```

* run yolo demo (for python)

```
$ pip install -r requirements.txt 
$ python yolov5.py 
```

***


# Reference
 - [Original Code](https://github.com/ronghuaiyang/arcface-pytorch)
 - [Paper](https://arxiv.org/abs/1801.07698)
