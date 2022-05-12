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

    model_size = 32
    model_name = "glean"

    cfg_file = "../models/glean/configs/{}.cfg".format(model_name)
    weight_file = "../models/glean/weights/{}.weights".format(model_name)
    engine_file = "../models/glean/engines/{}.bin".format(model_name)
    log_file = "../models/glean/logs/{}.log".format(model_name)

    extend_param = \
        "MODEL_NAME={} BATCH_SIZE={} ENGINE_SERIALIZE={} " \
        "MODEL_SIZE={},{} " \
        "WEIGHT_FILE={} ENGINE_FILE={} LOG_FILE={}".format(
            model_name, batch_size, engine_serialize,
            model_size, model_size,
            weight_file, engine_file, log_file
        )

    # Create SoyNet Handle
    handle = initSoyNet(cfg_file, extend_param)

    # WarmingUp SoyNet
    inference(handle)

    # Read Test Data
    img = cv.imread("../data/bird_32x32.png")

    # Resize Image
    resized_img = cv.resize(img, (model_size, model_size))

    # Create Output Variable
    output = np.zeros((batch_size, 3, model_size * 8, model_size * 8), dtype=np.float32)

    # FeedData
    feedData(handle, resized_img)

    # Inference
    inference(handle)

    # GetOutput
    getOutput(handle, output)

    # Post-Processing
    print("Output Shape: {}".format(output.shape))

    # destroy SoyNet handle
    freeSoyNet(handle)

    # View Result
    ViewResult(img, output, 'Glean')
