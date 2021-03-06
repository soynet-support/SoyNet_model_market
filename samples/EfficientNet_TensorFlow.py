import cv2 as cv
import sys
import numpy as np

sys.path.append('../')

from utils.ClassName import Imagenet1000
from include.SoyNet import *

if __name__ == "__main__":
    model_sizes = (224, 240, 260, 300, 380, 456, 528, 600, 800)
    version = {"B0": 0, "B1": 1, "B2": 2, "B3": 3, "B4": 4, "B5": 5, "B6": 6, "B7": 7, "L2": 8}
    class_names = Imagenet1000()

    # Variable for SoyNet
    model_code = version["B0"]
    batch_size = 1
    engine_serialize = 0  # 1: Create Engine For SoyNet, 0: Use of Engine generated

    class_count = len(class_names)
    model_height, model_width = model_sizes[model_code], model_sizes[model_code]

    model_name = "efficientnet-b{}".format(model_code)
    cfg_file = "../models/EfficientNet_TensorFlow/configs/{}.cfg".format(model_name)
    weight_file = "../models/EfficientNet_TensorFlow/weights/{}_noisy-student.weights".format(model_name)
    engine_file = "../models/EfficientNet_TensorFlow/engines/{}.bin".format(model_name)
    log_file = "../models/EfficientNet_TensorFlow/logs/{}.log".format(model_name)

    extend_param = \
        "MODEL_NAME={} BATCH_SIZE={} ENGINE_SERIALIZE={} CLASS_COUNT={} " \
        "MODEL_SIZE={},{} " \
        "WEIGHT_FILE={} ENGINE_FILE={} LOG_FILE={}".format(
            model_name, batch_size, engine_serialize, class_count,
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
    img = cv.imread("../data/panda5.jpg")

    # Resize Image
    resized_img = cv.resize(img, (model_width, model_height))

    # Create Output Variable
    output = np.zeros((batch_size * class_count), dtype=np.float32)

    # Use feedData, inference, getOutput to inference.
    # If a handle is already created, these can be used repeatedly.
    # FeedData
    feedData(handle, resized_img)

    # Inference
    inference(handle)

    # GetOutput
    getOutput(handle, output)

    # Post-Processing
    max_idx = np.argmax(output)

    print("Max Value: {} \nMax Index: {} \nClass Name: {}".format(output[max_idx], max_idx, class_names[max_idx]))
    
    # destroy SoyNet handle
    # freeSoyNet() removes the handle.
    # If you want to use the model again after removing the handle, create the handle again.
    freeSoyNet(handle)
