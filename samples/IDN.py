import cv2 as cv
import sys
import numpy as np

sys.path.append('../')

from include.SoyNet import *

if __name__ == "__main__":

    # Variable for SoyNet
    batch_size = 1
    engine_serialize = 0  # 1: Create Engine For SoyNet, 0: Use of Engine generated
    scale = 2

    model_height, model_width = 256, 256
    model_name = "IDN_pytorch"

    cfg_file = "../models/IDN/configs/{}.cfg".format(model_name)
    weight_file = "../models/IDN/weights/{}.weights".format(model_name)
    engine_file = "../models/IDN/engines/{}.bin".format(model_name)
    log_file = "../models/IDN/logs/{}.log".format(model_name)

    extend_param = \
        "MODEL_NAME={} BATCH_SIZE={} ENGINE_SERIALIZE={} SCALE={} " \
        "MODEL_SIZE={},{} " \
        "WEIGHT_FILE={} ENGINE_FILE={} LOG_FILE={}".format(
            model_name, batch_size, engine_serialize, scale,
            model_height, model_width,
            weight_file, engine_file, log_file
        )

    # Create SoyNet Handle
    handle = initSoyNet(cfg_file, extend_param)

    # WarmingUp SoyNet
    inference(handle)

    # Read Test Data
    img = cv.imread("../data/baby_x2_bicubic.jpg")
    
    # Resize Image
    resized_img = cv.resize(img, (model_width, model_height))

    # Create Output Variable
    P = model_height * scale
    Q = model_width * scale
    output = np.zeros((batch_size, resized_img.shape[2], P, Q), dtype=np.float32)

    # FeedData
    feedData(handle, resized_img)

    # Inference
    inference(handle)

    # GetOutput
    getOutput(handle, output)

    # Post-Processing
    # [N, C, H, W] -> [N, H, W, C]
    for n in range(batch_size):
        result = np.transpose(output[n], (1, 2, 0))

    print("Output Shape: {}".format(output.shape))

    # destroy SoyNet handle
    freeSoyNet(handle)
