import cv2 as cv
import sys
import numpy as np

sys.path.append('../')

from include.SoyNet import *
from utils.utils import ViewResult

if __name__ == "__main__":

    # Variable for SoyNet
    batch_size = 1
    engine_serialize = 0  # 1: Create Engine For SoyNet, 0: Use of Engine generated
    ngf = 64

    model_height, model_width = 720, 1280
    model_name = "pix2pix"

    cfg_file = "../models/pix2pix/configs/{}.cfg".format(model_name)
    weight_file = "../models/pix2pix/weights/{}.weights".format(model_name)
    engine_file = "../models/pix2pix/engines/{}.bin".format(model_name)
    log_file = "../models/pix2pix/logs/{}.log".format(model_name)

    extend_param = \
        "MODEL_NAME={} BATCH_SIZE={} ENGINE_SERIALIZE={} NGF={} " \
        "MODEL_SIZE={},{} " \
        "WEIGHT_FILE={} ENGINE_FILE={} LOG_FILE={}".format(
            model_name, batch_size, engine_serialize, ngf,
            model_height, model_width,
            weight_file, engine_file, log_file
        )

    # Create SoyNet Handle
    handle = initSoyNet(cfg_file, extend_param)

    # WarmingUp SoyNet
    inference(handle)

    # Read Test Data
    img = cv.imread("../data/NY_720x1280.jpg")
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Resize Image
    resized_img = cv.resize(img, (model_width, model_height))
    resized_img = np.expand_dims(resized_img, axis=2)

    # Create Output Variable
    output = np.zeros((batch_size, resized_img.shape[2], model_height, model_width), dtype=np.float32)

    # FeedData
    feedData(handle, resized_img)

    # Inference
    inference(handle)

    # GetOutput
    getOutput(handle, output)

    print("Output Shape: {}".format(output.shape))
    
    # destroy SoyNet handle
    freeSoyNet(handle)

    # View Result
    ViewResult(img, output, 'Pix2Pix')
