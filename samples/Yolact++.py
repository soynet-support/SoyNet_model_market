import cv2 as cv
import sys
import numpy as np

sys.path.append('../')

from utils.ClassName import COCO_80
from include.SoyNet import *

if __name__ == "__main__":

    class_names = COCO_80()

    # Variable for SoyNet
    batch_size = 1
    engine_serialize = 0  # 1: Create Engine For SoyNet, 0: Use of Engine generated

    nms_count = 100
    class_count = len(class_names)
    input_height, input_width = 720, 1280

    model_name = "yolactpp-resnet50"

    cfg_file = "../models/Yolact++/configs/{}.cfg".format(model_name)
    weight_file = "../models/Yolact++/weights/{}.weights".format(model_name)
    engine_file = "../models/Yolact++/engines/{}.bin".format(model_name)
    log_file = "../models/Yolact++/logs/{}.log".format(model_name)

    extend_param = \
        "MODEL_NAME={} BATCH_SIZE={} ENGINE_SERIALIZE={} CLASS_COUNT={} NMS_COUNT={} " \
        "INPUT_SIZE={},{} " \
        "WEIGHT_FILE={} ENGINE_FILE={} LOG_FILE={}".format(
            model_name, batch_size, engine_serialize, class_count, nms_count,
            input_height, input_width,
            weight_file, engine_file, log_file
        )

    # Create SoyNet Handle
    handle = initSoyNet(cfg_file, extend_param)

    # WarmingUp SoyNet
    inference(handle)

    # Read Test Data
    img = cv.imread("../data/NY_720x1280.jpg")

    # Resize Image
    resized_img = cv.resize(img, (input_width, input_height))

    # Create Output Variable
    data_type = np.dtype([("rip", [("x1", c_float), ("y1", c_float), ("x2", c_float), ("y2", c_float),
                          ("obj_id", c_float), ("prob", c_float)], nms_count), ("masked_image", c_uint8, (input_height, input_width, 3))])

    output=np.zeros(batch_size, dtype=data_type)
    # FeedData
    feedData(handle, resized_img)

    # Inference
    inference(handle)

    # GetOutput
    getOutput(handle, output)

    print("get")

    # Post-Processing
    for b_idx in range(batch_size):
        print("\nBatch_Num: {}".format(b_idx))
        rip, masked_image = output[b_idx]
        for n_idx in range(nms_count):
            x1, y1, x2, y2, obj_id, prob = rip[n_idx]
            print("NMS_Num: {} \nx1: {} \ny1: {} \nx2: {} \ny2: {} \nobj_id: {} \nprob: {} \nClass_name: {}\n".format(
                n_idx, x1, y1, x2, y2, obj_id, prob, class_names[int(obj_id)]))

        print("Output Shape: {}".format(masked_image.shape))                # [N, H, W, C]

    # destroy SoyNet handle
    freeSoyNet(handle)
