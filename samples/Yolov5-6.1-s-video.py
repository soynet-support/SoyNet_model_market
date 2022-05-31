import cv2 as cv
import sys
import numpy as np
import time
import argparse

sys.path.append('../')

from utils.ClassName import COCO_80
from utils.utils import ViewResult, ViewFPS
from include.SoyNet import *

parser = argparse.ArgumentParser(description="Set Value")
parser.add_argument('-t', '--threshold',
                    required=False,
                    type=float,
                    default=0.45,
                    help="Set Threshold")

if __name__ == "__main__":

    args = parser.parse_args()

    class_names = COCO_80()

    # Variable for SoyNet
    batch_size = 1
    engine_serialize = 0  # 1: Create Engine For SoyNet, 0: Use of Engine generated

    region_count = 1000
    nms_count = 100
    class_count = len(class_names)
    input_height, input_width = 720, 1280
    model_size = 640

    model_name = "yolov5s"

    cfg_file = "../models/Yolov5-6.1-s/configs/{}.cfg".format(model_name)
    weight_file = "../models/Yolov5-6.1-s/weights/{}.weights".format(model_name)
    engine_file = "../models/Yolov5-6.1-s/engines/{}.bin".format(model_name)
    log_file = "../models/Yolov5-6.1-s/logs/{}.log".format(model_name)

    extend_param = \
        "MODEL_NAME={} BATCH_SIZE={} ENGINE_SERIALIZE={} CLASS_COUNT={} NMS_COUNT={} REGION_COUNT={} " \
        "INPUT_SIZE={},{} MODEL_SIZE={},{} " \
        "WEIGHT_FILE={} ENGINE_FILE={} LOG_FILE={}".format(
            model_name, batch_size, engine_serialize, class_count, nms_count, region_count,
            input_height, input_width, model_size, model_size,
            weight_file, engine_file, log_file
        )

    # Create SoyNet Handle
    # initSoyNet() is used to create the engine of the model and the handle of SoyNet.
    # Use only for the first run.Once you have created a handle, you do not need to recreate handle.
    handle = initSoyNet(cfg_file, extend_param)

    # Read Test Data
    cap = cv.VideoCapture("../data/NY.mkv")

    # Create Output Variable
    data_type = np.dtype([("x1", c_float), ("y1", c_float), ("x2", c_float), ("y2", c_float),
                          ("obj_id", c_int), ("prob", c_float)])
    output = np.zeros(batch_size * nms_count, dtype=data_type)

    # Use feedData, inference, getOutput to inference.
    # If a handle is already created, these can be used repeatedly.
    total_fps = 0
    img_count = 0
    if cap.isOpened():
        while True:
            ret, img = cap.read()
            if ret:

                # Resize Image
                resized_img = cv.resize(img, (input_width, input_height))

                # Start Time measurement
                start = time.time()
                # FeedData
                feedData(handle, resized_img)

                # Inference
                inference(handle)

                # GetOutput
                getOutput(handle, output)
                # End Time measurement
                end = time.time()

                # Post-Processing
                for n_idx in range(nms_count):
                    x1, y1, x2, y2, obj_id, prob = output[n_idx]
                    if prob >= args.threshold:
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cv.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv.putText(img, class_names[obj_id], (x1, y1 - 3), 1, 1.5, (255, 0, 0), 1, cv.LINE_AA)

                        if x1 != 0 and y1 != 0 and x2 != 0 and y2 != 0:
                            print(
                                "NMS_Num: {} \nx1: {} \ny1: {} \nx2: {} \ny2: {} \n"
                                "obj_id: {} \nprob: {} \nClass_name: {}\n".format(
                                    n_idx, x1, y1, x2, y2, obj_id, prob, class_names[obj_id]))

                # Write FPS
                img, fps = ViewFPS(img, start, end, input_height, "Red")

                # Sum for Total FPS
                total_fps += fps
                img_count += 1

                cv.imshow('Test', img)
                if cv.waitKey(1) == ord('q'):
                    cv.destroyWindow('Test')
                    break
            else:
                break
    else:
        print("Video file does not exist.")

    # Print Average FPS
    print("Average FPS : {}".format(total_fps / img_count))

    # destroy SoyNet handle
    # freeSoyNet() removes the handle.
    # If you want to use the model again after removing the handle, create the handle again.
    freeSoyNet(handle)
