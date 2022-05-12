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
    inst_id = 0

    model_height, model_width = 1280, 1920
    model_name = "f_anogan"

    cfg_file = "../models/FAnoGan/configs/{}.cfg".format(model_name)
    weight_file = "../models/FAnoGan/weights/{}.weights".format(model_name)
    engine_file = "../models/FAnoGan/engines/{}.bin".format(model_name)
    log_file = "../models/FAnoGan/logs/{}.log".format(model_name)

    extend_param = \
        "MODEL_NAME={} BATCH_SIZE={} ENGINE_SERIALIZE={} INST_NAME=thread_{} " \
        "MODEL_SIZE={},{} " \
        "WEIGHT_FILE={} ENGINE_FILE={} LOG_FILE={}".format(
            model_name, batch_size, engine_serialize, inst_id,
            model_height, model_width,
            weight_file, engine_file, log_file
        )

    # Create SoyNet Handle
    handle = initSoyNet(cfg_file, extend_param)

    # WarmingUp SoyNet
    inference(handle)

    # Read Test Data
    img = cv.imread("../data/1280x1920.jpg")

    rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Resize Image
    resized_img = cv.resize(rgb, (model_width, model_height))

    # Create Output Variable
    output = np.zeros((batch_size, resized_img.shape[2], model_height, model_width), dtype=np.float32)

    # FeedData
    feedData(handle, resized_img)

    # Inference
    inference(handle)

    # GetOutput
    getOutput(handle, output)

    print("Output Shape: {}".format(output.shape))      # [N, C, H, W]
    
    # destroy SoyNet handle
    freeSoyNet(handle)

    # View Result
    ViewResult(img, output, 'FAnoGan')
