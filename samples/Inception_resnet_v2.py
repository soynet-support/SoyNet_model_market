import cv2 as cv
import sys
import numpy as np

sys.path.append('../')

from utils.ClassName import Imagenet1000_bg
from include.SoyNet import *

if __name__ == "__main__":

    class_names = Imagenet1000_bg()

    # Variable for SoyNet
    batch_size = 1
    engine_serialize = 1  # 1: Create Engine For SoyNet, 0: Use of Engine generated

    class_count = len(class_names)
    model_height, model_width = 299, 299

    model_name = "inception-resnet-v2"
    cfg_file = "../models/Inception_resnet_v2/configs/{}.cfg".format(model_name)
    weight_file = "../models/Inception_resnet_v2/weights/{}.weights".format(model_name)
    engine_file = "../models/Inception_resnet_v2/engines/{}.bin".format(model_name)
    log_file = "../models/Inception_resnet_v2/logs/{}.log".format(model_name)

    extend_param = \
        "MODEL_NAME={} BATCH_SIZE={} ENGINE_SERIALIZE={} CLASS_COUNT={} " \
        "WEIGHT_FILE={} ENGINE_FILE={} LOG_FILE={}".format(
            model_name, batch_size, engine_serialize, class_count,
            weight_file, engine_file, log_file
        )

    # Create SoyNet Handle
    handle = initSoyNet(cfg_file, extend_param)

    # WarmingUp SoyNet
    inference(handle)

    # Read Test Data
    img = cv.imread("../data/panda5.jpg")
    if img is None:
        print("Image is None!")
        sys.exit()

    # Resize Image
    resized_img = cv.resize(img, (model_width, model_height))

    # Create Output Variable
    output = np.zeros((batch_size * class_count), dtype=np.float32)

    # FeedData
    feedData(handle, resized_img)

    # Inference
    inference(handle)

    # GetOutput
    getOutput(handle, output)

    # Post-Processing