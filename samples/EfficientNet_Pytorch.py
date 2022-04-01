import cv2 as cv
import sys
import numpy as np

from utils.ClassName import Imagenet1000
from include.SoyNet import *

if __name__ == "__main__":
    model_sizes = (224, 240, 260, 300, 380, 456, 528, 600, 800)
    version = {"B0": 0, "B1": 1, "B2": 2, "B3": 3, "B4": 4, "B5": 5, "B6": 6, "B7": 7}
    class_names = Imagenet1000()

    # Variable for SoyNet
    model_code = version["B5"]
    batch_size = 1
    engine_serialize = 0  # 1: Create Engine For SoyNet, 0: Use of Engine generated

    class_count = len(class_names)
    model_height, model_width = model_sizes[model_code], model_sizes[model_code]

    model_name = "efficientnet-b{}_pt".format(model_code)
    cfg_file = "../models/EfficientNet_pytorch/configs/{}.cfg".format(model_name)
    weight_file = "../models/EfficientNet_pytorch/weights/{}.weights".format(model_name)
    engine_file = "../models/EfficientNet_pytorch/engines/{}.bin".format(model_name)
    log_file = "../models/EfficientNet_pytorch/logs/{}.log".format(model_name)

    extend_param = \
        "MODEL_NAME={} BATCH_SIZE={} ENGINE_SERIALIZE={} CLASS_COUNT={} " \
        "MODEL_SIZE={},{} " \
        "WEIGHT_FILE={} ENGINE_FILE={} LOG_FILE={}".format(
            model_name, batch_size, engine_serialize, class_count,
            model_height, model_width,
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
    max_idx = np.argmax(output)

    print("Max Value: {} \nMax Index: {} \nClass Name: {}".format(output[max_idx], max_idx, class_names[max_idx]))
