import cv2 as cv
import sys
import numpy as np

sys.path.append('../')

from utils.utils import ViewResult
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
    # initSoyNet() is used to create the engine of the model and the handle of SoyNet.
    # Use only for the first run.Once you have created a handle, you do not need to recreate handle.
    handle = initSoyNet(cfg_file, extend_param)

    # WarmingUp SoyNet
    inference(handle)

    # Read Test Data
    img = cv.imread("../data/NY_720x1280.jpg")

    # Resize Image
    resized_img = cv.resize(img, (model_width, model_height))

    # Create Output Variable
    output = np.zeros((batch_size, model_height, model_width, resized_img.shape[2]), dtype=np.float32)

    # Use feedData, inference, getOutput to inference.
    # If a handle is already created, these can be used repeatedly.
    # FeedData
    feedData(handle, resized_img)

    # Inference
    inference(handle)

    # GetOutput
    getOutput(handle, output)

    # Post-Processing
    print("Output Shape: {}".format(output.shape))
    
    # destroy SoyNet handle
    # freeSoyNet() removes the handle.
    # If you want to use the model again after removing the handle, create the handle again.
    freeSoyNet(handle)

    # View Result
    ViewResult(img, output, 'CycleGan')
