import cv2 as cv
import sys
import numpy as np

sys.path.append('../')

from include.SoyNet import *

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

    # WarmingUp SoyNet
    inference(handle)

    # Read Test Data
    img = cv.imread("../data/zidane.jpg")
    if img is None:
        print("Image is None!")
        sys.exit()

    # Resize Image
    resized_img = cv.resize(img, (input_width, input_height))

    # Create Output Variable
    output = np.zeros((batch_size, nms_count * (6 + 17 * 3)), dtype=np.float32)

    # FeedData
    feedData(handle, resized_img)

    # Inference
    inference(handle)

    # GetOutput
    getOutput(handle, output)

    # Post-Processing
    for b_idx in range(batch_size):
        result = output[b_idx]
        rip = result[: nms_count * 6]
        keypoint = result[nms_count * 6:]
        for n_idx in range(nms_count):
            x1, y1, x2, y2, obj_id, prob = rip[n_idx * 6 : (n_idx + 1) * 6]  # 6 + 17 * 3
            nose_x, nose_y, nose_prob, r_eye_x, r_eye_y, r_eye_prob, \
            l_eye_x, l_eye_y, l_eye_prob, r_ear_x, r_ear_y, r_ear_prob, l_ear_x, l_ear_y, l_ear_prob, \
            r_shoulder_x, r_shoulder_y, r_shoulder_prob, l_shoulder_x, l_shoulder_y, l_shoulder_prob, \
            r_elbow_x, r_elbow_y, r_elbow_prob, l_elbow_x, l_elbow_y, l_elbow_prob, r_wrist_x, r_wrist_y, r_wrist_prob,\
            l_wrist_x, l_wrist_y, l_wrist_prob, r_pelvis_x, r_pelvis_y, r_pelvis_prob, \
            l_pelvis_x, l_pelvis_y, l_pelvis_prob, r_knee_x, r_knee_y, r_knee_prob, l_knee_x, l_knee_y, l_knee_prob, \
            r_foot_x, r_foot_y, r_foot_prob, l_foot_x, l_foot_y, l_foot_prob \
                = keypoint[n_idx * 51: (n_idx + 1) * 51]

    print("Output Shape: {}".format(output.shape))
