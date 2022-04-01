import cv2 as cv
import sys
import numpy as np
from utils.utils import CreateNetsizeImage

from include.SoyNet import *


if __name__ == "__main__":

    # Variable for SoyNet
    batch_size = 1
    engine_serialize = 0  # 1: Create Engine For SoyNet, 0: Use of Engine generated

    nms_count = 50
    input_height, input_width = 200, 200

    model_name = "openpose-yolov4"
    cfg_file = "../models/Openpose-Darknet/configs/{}.cfg".format(model_name)
    weight_file = "../models/Openpose-Darknet/weights/{}.weights".format(model_name)
    engine_file = "../models/Openpose-Darknet/engines/{}.bin".format(model_name)
    log_file = "../models/Openpose-Darknet/logs/{}.log".format(model_name)

    extend_param = \
        "MODEL_NAME={} BATCH_SIZE={} ENGINE_SERIALIZE={} " \
        "INPUT_SIZE={},{} " \
        "WEIGHT_FILE={} ENGINE_FILE={} LOG_FILE={}".format(
            model_name, batch_size, engine_serialize,
            input_height, input_width,
            weight_file, engine_file, log_file
        )

    # Create SoyNet Handle
    handle = initSoyNet(cfg_file, extend_param)

    # WarmingUp SoyNet
    inference(handle)

    # Read Test Data
    img = cv.imread("../data/NY_720x1280.jpg")
    if img is None:
        print("Image is None!")
        sys.exit()
    neww, newh = CreateNetsizeImage(img, 200, 200, 0)
    resized_img = np.zeros(shape=(200, 200, 3), dtype=np.uint8)
    resized_img[:newh, :neww, :] = cv.resize(img, (neww, newh))

    # Create Output Variable
    output = np.zeros(25 * 25 * 57, dtype=np.float32)

    # FeedData
    feedData(handle, resized_img)

    # Inference
    inference(handle)

    # GetOutput
    getOutput(handle, output)

    # Post-Processing
    print("Output Shape: {}".format(output.shape))
