import cv2 as cv
import sys
import numpy as np
import argparse
import gradio as gr

sys.path.append('../')

from include.SoyNet import *
from utils.utils import MakeMultiple32

def predict(img):
    # Create Output Variable
    data_type = np.dtype([("x1", c_float), ("y1", c_float), ("x2", c_float), ("y2", c_float),
                          ("prob", c_float),
                          ("r_eye1", c_float), ("r_eye2", c_float),
                          ("l_eye1", c_float), ("l_eye2", c_float),
                          ("nose1", c_float), ("nose2", c_float),
                          ("r_mouth1", c_float), ("r_mouth2", c_float),
                          ("l_mouth1", c_float), ("l_mouth2", c_float)])
    output = np.zeros(batch_size * nms_count * 15, dtype=data_type)

    # Use feedData, inference, getOutput to inference.
    # If a handle is already created, these can be used repeatedly.
    feedData(handle, img)
    inference(handle)
    getOutput(handle, output)

    # apply predicted result to original image
    for n_idx in range(nms_count):
            result = output[n_idx]
            cv.rectangle(img, (int(result[0]), int(result[1])), (int(result[2]), int(result[3])), (0, 0, 255), 2)
            for idx in range(5, 15, 2):
                cv.circle(img, (int(result[idx]), int(result[idx + 1])), 5, (0, 255, 255), -1)
    return img

if __name__ == "__main__":
    class_names = (0, 1)

    # Variable for SoyNet
    batch_size = 1
    engine_serialize = 0  # 1: Create Engine For SoyNet, 0: Use of Engine generated

    region_count = 1000
    nms_count = 10
    class_count = len(class_names)
    model_height, model_width = MakeMultiple32(960), MakeMultiple32(1280)

    model_name = "retina_face_r50"
    cfg_file = "../models/RetinaFace/configs/{}.cfg".format(model_name)
    weight_file = "../models/RetinaFace/weights/{}.weights".format(model_name)
    engine_file = "../models/RetinaFace/engines/{}.bin".format(model_name)
    log_file = "../models/RetinaFace/logs/{}.log".format(model_name)

    extend_param = \
        "MODEL_NAME={} BATCH_SIZE={} ENGINE_SERIALIZE={} CLASS_COUNT={} NMS_COUNT={} REGION_COUNT={} " \
        "MODEL_SIZE={},{} " \
        "WEIGHT_FILE={} ENGINE_FILE={} LOG_FILE={}".format(
            model_name, batch_size, engine_serialize, class_count, nms_count, region_count,
            model_height, model_width,
            weight_file, engine_file, log_file
        )

    # Create SoyNet Handle
    # initSoyNet() is used to create the engine of the model and the handle of SoyNet.
    # Use only for the first run.Once you have created a handle, you do not need to recreate handle.
    handle = initSoyNet(cfg_file, extend_param)

    demo = gr.Interface(
        fn=predict,
        inputs=gr.inputs.Image(shape=(model_width, model_height)),  
        outputs="image",
        title='SoyNet RetinaFace Demo',
        description='This is RetinaFace demo using SoyNet & Gradio',
        examples=['examples/t1_1280x960.jpg'],
    )
    try:
        demo.launch()
    except KeyboardInterrupt:
        # destroy SoyNet handle
        freeSoyNet(handle)    
