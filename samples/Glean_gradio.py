
import cv2 as cv
import sys
import numpy as np
import argparse
import gradio as gr

sys.path.append('../')

from include.SoyNet import *
from utils.utils import ViewResult

from utils.ClassName import COCO_80
class_names = COCO_80()

def predict(img):
    # Resize Image
    resized_img = cv.resize(img, (model_size, model_size))

    # Create Output Variable
    output = np.zeros((batch_size, 3, model_size * 8, model_size * 8), dtype=np.float32)

    # Use feedData, inference, getOutput to inference.
    # If a handle is already created, these can be used repeatedly.
    feedData(handle, resized_img)
    inference(handle)
    getOutput(handle, output)
    
    # post processing for gradio output image 
    output = np.reshape(output, (output.shape[2], output.shape[3], output.shape[1]))
    output = cv.cvtColor(output, cv.COLOR_BGR2RGB)
    output /= output.max()
    
    return output

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
            weight_file, engine_file, log_file)

    # Create SoyNet Handle
    # initSoyNet() is used to create the engine of the model and the handle of SoyNet.
    # Use only for the first run.Once you have created a handle, you do not need to recreate handle.
    handle = initSoyNet(cfg_file, extend_param)

    demo = gr.Interface(
        fn=predict,
        inputs="image",
        outputs="image",
        title='SoyNet Glean Demo',
        description='This is gradio demo using SoyNet inference engine.',
        examples=['examples/bird_32x32.png'],
    )
    try:
        demo.launch()
    except KeyboardInterrupt:
        # destroy SoyNet handle
        freeSoyNet(handle)    
