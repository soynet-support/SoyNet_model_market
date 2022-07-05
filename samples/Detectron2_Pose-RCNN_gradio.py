import cv2 as cv
import sys
import numpy as np
import gradio as gr

sys.path.append('../')

from utils.utils import ViewResult
from include.SoyNet import *


def predict(img):
    # resize image
    resized_img = cv.resize(img, (input_width, input_height))
    hs, ws = img.shape[0]/input_height, img.shape[1]/input_width
    
    # Create Output Variable
    output = np.zeros((batch_size, nms_count * (6 + 17 * 3)), dtype=np.float32)

    # inference.
    feedData(handle, resized_img)
    inference(handle)
    getOutput(handle, output)

    # apply predicted result to original image
    for b_idx in range(batch_size):
        result = output[b_idx]
        rip = result[: nms_count * 6]
        keypoint = result[nms_count * 6:]
        for n_idx in range(nms_count):
            x1, y1, x2, y2, obj_id, prob = rip[n_idx * 6: (n_idx + 1) * 6]  # 6 + 17 * 3
            # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            x1, x2 = int(x1 * ws), int(x2 * ws)
            y1, y2 = int(y1 * hs), int(y2 * hs)
            cv.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            for k_idx in range(n_idx * 51, (n_idx + 1) * 51, 3):
                cv.circle(img, (int(keypoint[k_idx]* ws), int(keypoint[k_idx + 1]* hs)), 5, (0, 255, 255), -1)
    return img

if __name__ == "__main__":

    # Variable for SoyNet
    batch_size = 1
    engine_serialize = 0  # 1: Create Engine For SoyNet, 0: Use of Engine generated

    region_count = 1000
    nms_count = 10
    class_count = 1
    keypoint_count = 17
    input_height, input_width = 720, 1280

    model_name = "pose-r101-fpn-detectron2"

    cfg_file = "../models/Detectron2_Pose-RCNN/configs/{}.cfg".format(model_name)
    weight_file = "../models/Detectron2_Pose-RCNN/weights/{}.weights".format(model_name)
    engine_file = "../models/Detectron2_Pose-RCNN/engines/{}.bin".format(model_name)
    log_file = "../models/Detectron2_Pose-RCNN/logs/{}.log".format(model_name)

    extend_param = \
        "MODEL_NAME={} BATCH_SIZE={} ENGINE_SERIALIZE={} CLASS_COUNT={} KEYPOINT_COUNT={} NMS_COUNT={} " \
        "INPUT_SIZE={},{} " \
        "WEIGHT_FILE={} ENGINE_FILE={} LOG_FILE={}".format(
            model_name, batch_size, engine_serialize, class_count, keypoint_count, nms_count,
            input_height, input_width,
            weight_file, engine_file, log_file
        )

    # Create SoyNet Handle
    handle = initSoyNet(cfg_file, extend_param)

    demo = gr.Interface(
        fn=predict,
        inputs='image',
        outputs='image',
        title='SoyNet Pose-RCNN2 Demo',
        description='This is Pose-RCNN demo using SoyNet & Gradio',
        examples=['examples/zidane.jpg','examples/fallen_01.jpg','examples/dance.jpg'],       #data/NY_720x1280.jpg
    )
    try:
        demo.launch()
    except KeyboardInterrupt:
        # destroy SoyNet handle
        freeSoyNet(handle)    

