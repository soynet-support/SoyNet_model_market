import cv2 as cv
import sys
import numpy as np
import gradio as gr

sys.path.append('../')

from include.SoyNet import *

from utils.ClassName import COCO_80

def predict(img):
    # Create Output Variable
    data_type = np.dtype([("rip", [("x1", c_float), ("y1", c_float), ("x2", c_float), ("y2", c_float),
                          ("obj_id", c_int), ("prob", c_float)], nms_count),
                          ("masked_image", c_uint8, (input_height, input_width, 3))])
    output = np.zeros(batch_size , dtype=data_type)
 
    # inference
    feedData(handle, img)
    inference(handle)
    getOutput(handle, output)

    # post processing predicted result to original image
    threshold = 0.45
    for b_idx in range(batch_size):
        rip, masked_img = output[b_idx]
        for r_idx in rip:
            x1, y1, x2, y2, obj_id, prob = r_idx
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv.rectangle(masked_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv.putText(masked_img, class_names[obj_id], (int(x1), int(y1) - 3), 1, 1.5, (255, 0, 0), 1, cv.LINE_AA)
    return masked_img

if __name__ == "__main__":
    class_names = COCO_80()

    # Variable for SoyNet
    batch_size = 1
    engine_serialize = 0  # 1: Create Engine For SoyNet, 0: Use of Engine generated

    nms_count = 100
    class_count = len(class_names)
    input_height, input_width = 720, 1280
    model_height, model_width = 512, 512

    model_name = "maskrcnn-r101-fpn-detectron2" # maskrcnn-x101-fpn-detectron2

    cfg_file = "../models/Detectron2_Mask-RCNN/configs/{}.cfg".format(model_name)
    weight_file = "../models/Detectron2_Mask-RCNN/weights/{}.weights".format(model_name)
    engine_file = "../models/Detectron2_Mask-RCNN/engines/{}.bin".format(model_name)
    log_file = "../models/Detectron2_Mask-RCNN/logs/{}.log".format(model_name)

    extend_param = \
        "MODEL_NAME={} BATCH_SIZE={} ENGINE_SERIALIZE={} CLASS_COUNT={} NMS_COUNT={} " \
        "INPUT_SIZE={},{} MODEL_SIZE={},{} " \
        "WEIGHT_FILE={} ENGINE_FILE={} LOG_FILE={}".format(
            model_name, batch_size, engine_serialize, class_count, nms_count,
            input_height, input_width, model_height, model_width,
            weight_file, engine_file, log_file
        )

    # Create SoyNet Handle
    # initSoyNet() is used to create the engine of the model and the handle of SoyNet.
    # Use only for the first run.Once you have created a handle, you do not need to recreate handle.
    handle = initSoyNet(cfg_file, extend_param)

    demo = gr.Interface(
        fn=predict,
        inputs=gr.inputs.Image(shape=(input_width, input_height)),  
        outputs="image",
        title='SoyNet Detectron2 MaskRCNN Demo',
        description='This is Detectron2 MaskRCNN demo using SoyNet & Gradio',
        examples=['examples/NY_01.png','examples/NY_02.png'],       #data/NY_720x1280.jpg
    )
    try:
        demo.launch()
    except KeyboardInterrupt:
        # destroy SoyNet handle
        freeSoyNet(handle)    
