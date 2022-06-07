import cv2 as cv
import sys
import numpy as np
import argparse
import gradio as gr

sys.path.append('../')

from include.SoyNet import *

from utils.ClassName import COCO_80
class_names = COCO_80()

def predict(img):
    # Resize Image
    resized_img = cv.resize(img, (input_width, input_height))

    # Create Output Variable
    data_type = np.dtype([("x1", c_float), ("y1", c_float), ("x2", c_float), ("y2", c_float),
                            ("obj_id", c_int), ("prob", c_float)])
    output = np.zeros(batch_size * nms_count, dtype=data_type)

    # Use feedData, inference, getOutput to inference.
    # If a handle is already created, these can be used repeatedly.
    feedData(handle, resized_img)
    inference(handle)
    getOutput(handle, output)

    # apply predicted result to original image
    threshold = 0.6
    for n_idx in range(nms_count):
        x1, y1, x2, y2, obj_id, prob = output[n_idx]
        if prob > threshold:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv.putText(img, class_names[obj_id], (x1, y1 - 3), 1, 1.5, (255, 0, 0), 1, cv.LINE_AA)
    return img

if __name__ == "__main__":
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

    demo = gr.Interface(
        fn=predict,
        inputs="image",
        outputs="image",
        title='SoyNet gradio Demo',
        description='This is gradio demo using SoyNet inference engine.',
        examples=['examples/NY_01.png','examples/NY_02.png'],
    )
    try:
        demo.launch()
    except KeyboardInterrupt:
        # destroy SoyNet handle
        freeSoyNet(handle)    
