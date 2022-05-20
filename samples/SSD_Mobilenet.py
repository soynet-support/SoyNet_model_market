import cv2 as cv
import sys
import numpy as np

sys.path.append('../')

from utils.ClassName import COCO_90
from utils.utils import ViewResult
from include.SoyNet import *

if __name__ == "__main__":
    class_names = COCO_90()

    # Variable for SoyNet
    batch_size = 1
    engine_serialize = 0  # 1: Create Engine For SoyNet, 0: Use of Engine generated

    region_count = 1000
    nms_count = 50
    class_count = len(class_names)
    model_height, model_width = 640, 640

    model_name = "ssd_mobilenet_v2_fpnlite_640x640"
    cfg_file = "../models/SSD_Mobilenet/configs/{}.cfg".format(model_name)
    weight_file = "../models/SSD_Mobilenet/weights/{}.weights".format(model_name)
    engine_file = "../models/SSD_Mobilenet/engines/{}.bin".format(model_name)
    log_file = "../models/SSD_Mobilenet/logs/{}.log".format(model_name)

    extend_param = \
        "MODEL_NAME={} BATCH_SIZE={} ENGINE_SERIALIZE={} CLASS_COUNT={} NMS_COUNT={} REGION_COUNT={} " \
        "MODEL_SIZE={},{} " \
        "WEIGHT_FILE={} ENGINE_FILE={} LOG_FILE={}".format(
            model_name, batch_size, engine_serialize, class_count, nms_count, region_count,
            model_height, model_width,
            weight_file, engine_file, log_file
        )

    # Create SoyNet Handle
    # initSoyNet() is used to create the engine of the model and the handle of SoyNet.
    # Use only for the first run.Once you have created a handle, you do not need to recreate handle.
    handle = initSoyNet(cfg_file, extend_param)

    # WarmingUp SoyNet
    inference(handle)

    # Read Test Data
    img = cv.imread("../data/t1_1280x960.jpg")

    # Resize Image
    resized_img = cv.resize(img, (model_width, model_height))

    # Create Output Variable
    data_type = np.dtype([("x1", c_float), ("y1", c_float), ("x2", c_float), ("y2", c_float),
                          ("obj_id", c_int), ("prob", c_float)])
    output = np.zeros(batch_size * nms_count, dtype=data_type)

    # Use feedData, inference, getOutput to inference.
    # If a handle is already created, these can be used repeatedly.
    # FeedData
    feedData(handle, resized_img)

    # Inference
    inference(handle)

    # GetOutput
    getOutput(handle, output)

    # Post-Processing
    for b_idx in range(batch_size):
        print("\nBatch_Num: {}".format(b_idx))
        for n_idx in range(nms_count):
            x1, y1, x2, y2, obj_id, prob = output[n_idx + b_idx * nms_count]
            print("NMS_Num: {} \nx1: {} \ny1: {} \nx2: {} \ny2: {} \nobj_id: {} \nprob: {} \nClass_name: {}\n".format(
                n_idx, x1, y1, x2, y2, obj_id, prob, class_names[obj_id]))

    # destroy SoyNet handle
    # freeSoyNet() removes the handle.
    # If you want to use the model again after removing the handle, create the handle again.
    freeSoyNet(handle)

    # View Result
    ViewResult(img, output, 'SSD_Mobilenet', nms=nms_count, batch=batch_size)
