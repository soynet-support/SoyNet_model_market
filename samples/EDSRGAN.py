import numpy as np
import matplotlib.pyplot as plt
import cv2

import sys
sys.path.append('../')

from include.SoyNet import *

if __name__ == "__main__":

    # Variable for SoyNet
    engine_serialize = 1 # 1: Create Engine For SoyNet, 0: Use of Engine generated

    batch_size = 1
    input_height, input_width = 540, 960

    scale = 4
    model_name = "edsr-16-x{}".format(scale)

    cfg_file = "../mgmt/configs/{}.cfg".format(model_name)
    weight_file = "../mgmt/weights/{}.weights".format(model_name)
    engine_file = "../mgmt/engines/{}.bin".format(model_name)
    log_file = "../mgmt/logs/{}.log".format(model_name)

    extend_param = \
        "MODEL_NAME={} BATCH_SIZE={} ENGINE_SERIALIZE={} " \
        "INPUT_SIZE={},{} " \
        "WEIGHT_FILE={} ENGINE_FILE={} LOG_FILE={}".format(
            model_name, batch_size, engine_serialize,
            input_height, input_width,
            weight_file, engine_file, log_file
        )

    # Create SoyNet Handle
    # initSoyNet() is used to create the engine of the model and the handle of SoyNet.
    # Use only for the first run.Once you have created a handle, you do not need to recreate handle.
    handle = initSoyNet(cfg_file, extend_param)

    # WarmingUp SoyNet
    inference(handle)

    # Read Test Data
    lr_img = cv2.imread("../data/test.jpg")
    lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)

    # Create Output Variable
    output = np.zeros((batch_size, (input_height * scale), (input_width * scale), 3), dtype=np.float32) # NHWC

    # Use feedData, inference, getOutput to inference.
    # If a handle is already created, these can be used repeatedly.
    # FeedData
    feedData(handle, lr_img)

    # Inference
    inference(handle)

    # GetOutput
    getOutput(handle, output)

    # Post-Processing
    output = np.clip(output, 0, 255)
    output = np.round(output)
    output = output.astype(np.uint8)

    # View Result
    ViewResult(img, output, 'EDSRGAN')
