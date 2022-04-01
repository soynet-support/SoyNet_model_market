import cv2 as cv
import sys
import numpy as np

from include.SoyNet import *
from utils.utils import MakeMultiple32

if __name__ == "__main__":
    class_names = (0, 1)

    # Variable for SoyNet
    batch_size = 1
    engine_serialize = 0  # 1: Create Engine For SoyNet, 0: Use of Engine generated

    region_count = 1000
    nms_count = 10
    class_count = len(class_names)
    model_height, model_width = MakeMultiple32(960), MakeMultiple32(1280)

    model_name = "retina_face_r50"
    cfg_file = "../models/RetinaFace/configs/{}.cfg".format(model_name)
    weight_file = "../models/RetinaFace/weights/{}.weights".format(model_name)
    engine_file = "../models/RetinaFace/engines/{}.bin".format(model_name)
    log_file = "../models/RetinaFace/logs/{}.log".format(model_name)

    extend_param = \
        "MODEL_NAME={} BATCH_SIZE={} ENGINE_SERIALIZE={} CLASS_COUNT={} NMS_COUNT={} REGION_COUNT={} " \
        "INPUT_SIZE={},{} " \
        "WEIGHT_FILE={} ENGINE_FILE={} LOG_FILE={}".format(
            model_name, batch_size, engine_serialize, class_count, nms_count, region_count,
            model_height, model_width,
            weight_file, engine_file, log_file
        )

    # Create SoyNet Handle
    handle = initSoyNet(cfg_file, extend_param)

    # WarmingUp SoyNet
    inference(handle)

    # Read Test Data
    img = cv.imread("../data/t1_1280x960.jpg")
    if img is None:
        print("Image is None!")
        sys.exit()

    # Resize Image
    resized_img = cv.resize(img, (model_width, model_height))

    # Create Output Variable
    data_type = np.dtype([("x1", c_float), ("y1", c_float), ("x2", c_float), ("y2", c_float),
                          ("prob", c_float),
                          ("r_eye1", c_float), ("r_eye2", c_float),
                          ("l_eye1", c_float), ("l_eye2", c_float),
                          ("nose1", c_float), ("nose2", c_float),
                          ("r_mouth1", c_float), ("r_mouth2", c_float),
                          ("l_mouth1", c_float), ("l_mouth2", c_float)])

    output = np.zeros(batch_size * nms_count * 15, dtype=data_type)

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
            x1, y1, x2, y2, prob, r_eye_x, r_eye_y, l_eye_x, l_eye_y,\
                nose_x, nose_y, r_mouth_x, r_mouth_y, l_mout_x, l_mout_y = output[n_idx + b_idx * nms_count]

            print("NMS_Num: {} \nx1: {} \ny1: {} \nx2: {} \ny2: {} \nprob: {} \n".format(
                n_idx, x1, y1, x2, y2, prob))
