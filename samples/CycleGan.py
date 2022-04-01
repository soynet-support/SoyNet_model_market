import cv2 as cv
import sys
import numpy as np

from include.SoyNet import *

if __name__ == "__main__":

    # Variable for SoyNet
    batch_size = 1
    engine_serialize = 0  # 1: Create Engine For SoyNet, 0: Use of Engine generated

    model_height, model_width = 720, 1280
    model_name = "cycleGAN"

    cfg_file = "../models/CycleGan/configs/{}.cfg".format(model_name)
    weight_file = "../models/CycleGan/weights/{}.weights".format(model_name)
    engine_file = "../models/CycleGan/engines/{}.bin".format(model_name)
    log_file = "../models/CycleGan/logs/{}.log".format(model_name)

    extend_param = \
        "MODEL_NAME={} BATCH_SIZE={} ENGINE_SERIALIZE={} " \
        "MODEL_SIZE={},{} " \
        "WEIGHT_FILE={} ENGINE_FILE={} LOG_FILE={}".format(
            model_name, batch_size, engine_serialize,
            model_height, model_width,
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

    # Resize Image
    resized_img = cv.resize(img, (model_width, model_height))

    # Create Output Variable
    output = np.zeros(batch_size * model_width * model_height * resized_img.shape[2], dtype=np.float32)

    # FeedData
    feedData(handle, resized_img)

    # Inference
    inference(handle)

    # GetOutput
    getOutput(handle, output)

    # Post-Processing
    print("Output Shape: {}".format(output.shape))
