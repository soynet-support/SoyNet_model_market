import cv2 as cv
import sys
import numpy as np

sys.path.append('../')

from utils.ClassName import COCO_80
from utils.utils import ViewResult
from include.SoyNet import *

if __name__ == "__main__":

    # Variable for SoyNet
    batch_size = 1
    engine_serialize = 0  # 1: Create Engine For SoyNet, 0: Use of Engine generated

    region_count = 1000
    nms_count = 10
    class_count = 16
    input_height, input_width = 720, 1280
    model_size = 640
    resize_ratio = float(model_size) / float(max(input_height, input_width))
    model_name = "yolov5sn-face"

    cfg_file = "../models/Yolov5_Face/configs/{}.cfg".format(model_name)
    weight_file = "../models/Yolov5_Face/weights/{}.weights".format(model_name)
    engine_file = "../models/Yolov5_Face/engines/{}.bin".format(model_name)
    log_file = "../models/Yolov5_Face/logs/{}.log".format(model_name)

    extend_param = \
        "MODEL_NAME={} BATCH_SIZE={} ENGINE_SERIALIZE={} CLASS_COUNT={} NMS_COUNT={} REGION_COUNT={} " \
        "INPUT_SIZE={},{} MODEL_SIZE={},{} RESIZE_RATIO={} " \
        "WEIGHT_FILE={} ENGINE_FILE={} LOG_FILE={}".format(
            model_name, batch_size, engine_serialize, class_count, nms_count, region_count,
            input_height * resize_ratio, input_width * resize_ratio, model_size, model_size, 1./resize_ratio,
            weight_file, engine_file, log_file
        )

    # Create SoyNet Handle
    handle = initSoyNet(cfg_file, extend_param)

    # WarmingUp SoyNet
    inference(handle)

    # Read Test Data
    img = cv.imread("../data/zidane.jpg")

    # Resize Image
    resized_img = cv.resize(img, (int(img.shape[1] * resize_ratio), int(img.shape[0] * resize_ratio)))

    # Create Output Variable
    data_type = np.dtype([("x1", c_float), ("y1", c_float), ("x2", c_float), ("y2", c_float),
                          ("prob", c_float),
                          ("r_eye1", c_float), ("r_eye2", c_float),
                          ("l_eye1", c_float), ("l_eye2", c_float),
                          ("nose1", c_float), ("nose2", c_float),
                          ("r_mouth1", c_float), ("r_mouth2", c_float),
                          ("l_mouth1", c_float), ("l_mouth2", c_float),
                          ("etc", c_float)])

    output = np.zeros(batch_size * nms_count, dtype=data_type)

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
            x1 = output[n_idx + b_idx * nms_count][0]           # Bounding Box
            y1 = output[n_idx + b_idx * nms_count][1]           # Bounding Box
            x2 = output[n_idx + b_idx * nms_count][2]           # Bounding Box
            y2 = output[n_idx + b_idx * nms_count][3]           # Bounding Box
            prob = output[n_idx + b_idx * nms_count][4]         # Probability
            r_eye_x = output[n_idx + b_idx * nms_count][5]      # Right Eye X
            r_eye_y = output[n_idx + b_idx * nms_count][6]      # Right Eye Y
            l_eye_x = output[n_idx + b_idx * nms_count][7]      # Left Eye X
            l_eye_y = output[n_idx + b_idx * nms_count][8]      # Left Eye Y
            nose_x = output[n_idx + b_idx * nms_count][9]       # Nose X
            nose_y = output[n_idx + b_idx * nms_count][10]      # Nose Y
            r_mouth_x = output[n_idx + b_idx * nms_count][11]   # Right Mouth X
            r_mouth_y = output[n_idx + b_idx * nms_count][12]   # Right Mouth Y
            l_mouth_x = output[n_idx + b_idx * nms_count][13]   # Left Mouth X
            l_mouth_y = output[n_idx + b_idx * nms_count][14]   # Left Mouth Y

            print("NMS_Num: {} \nx1: {} \ny1: {} \nx2: {} \ny2: {} \nprob: {} \n".format(
                n_idx, x1, y1, x2, y2, prob))

    # destroy SoyNet handle
    freeSoyNet(handle)

    # View Result
    ViewResult(img, output, 'Yolov5_face', nms=nms_count)
