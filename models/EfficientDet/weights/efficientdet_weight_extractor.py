# Core Author: Zylo117
# Script's Author: winter2897 

"""
Simple Inference Script of EfficientDet-Pytorch for detecting objects on webcam
"""
import time
import torch
import cv2
import numpy as np
from torch.backends import cudnn
from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, preprocess_video

# Video's path
video_src = 'NY.mkv'  # set int to use webcam, set str to read from a video file
#video_src = 0

compound_coef = 5
force_input_size = None  # set None to use default size

threshold = 0.2
iou_threshold = 0.2

use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True

obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush']

# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

# load model
model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list))
model.load_state_dict(torch.load(f'weights/efficientdet-d{compound_coef}.pth'))
model.requires_grad_(False)
model.eval()

# weigths extractor
import numpy as np

if 0:
    weights = model.state_dict()
    weight_list = [(key, value) for (key, value) in weights.items()]
    for idx in range(0, len(weight_list)):  #
        key, w = weight_list[idx]
        if "num_batches_tracked" in key:
            print(idx, "--------------------")
            continue
        if len(w.shape) == 2:
            print("transpose() \n")
            w = w.transpose(1, 0)
            w = w.cpu().data.numpy()
        else:
            w = w.cpu().data.numpy()
        print(0, idx, key, w.shape)

# weigths extractor d0
if 0:  # weight download, (0 -> off, 1 -> on)
    weight_path = "E:/DEV4/mgmt/weights/eff-det/efficientdet-d0.weights"
    with open(weight_path, 'wb') as f:
        weights = model.state_dict()
        weight_list = [(key, value) for (key, value) in weights.items()]
        dumy = np.array([0] * 10, dtype=np.float32)
        dumy.tofile(f)

        for idx in range(432, len(weight_list)):  #
            key, w = weight_list[idx]
            if "num_batches_tracked" in key:
                print(idx, "--------------------")
                continue
            if len(w.shape) == 2:
                print("transpose() \n")
                w = w.transpose(1, 0)
                w = w.cpu().data.numpy()
            else:
                w = w.cpu().data.numpy()
            w.tofile(f)
            print(0, idx, key, w.shape)

        index_l = [
                    # Bifpn0
                    93, 99,
                    72, 92, #first time0 순서 바꿈
                    0, 1, 8, 15,
                    1, 2, 16, 23,
                    2, 3, 24, 31,
                    3, 4, 32, 39,
                    100, 113, #first time1
                    4, 5, 40, 47,
                    5, 6, 48, 55,
                    6, 7, 56, 63,
                    7, 8, 64, 71,
                    # Bifpn1
                    114, 115, 122, 129,
                    115, 116, 130, 137,
                    116, 117, 138, 145,
                    117, 118, 146, 153,
                    118, 119, 154, 161,
                    119, 120, 162, 169,
                    120, 121, 170, 177,
                    121, 122, 178, 185,
                    # Bifpn2
                    186, 187, 194, 201,
                    187, 188, 202, 209,
                    188, 189, 210, 217,
                    189, 190, 218, 225,
                    190, 191, 226, 233,
                    191, 192, 234, 241,
                    192, 193, 242, 249,
                    193, 194, 250, 257,
                    # Regressor p3
                    258, 261, 267, 271,
                    261, 264, 272, 276,
                    264, 267, 277, 281,
                    342, 345,
                    # Regressor p4
                    258, 261, 282, 286,
                    261, 264, 287, 291,
                    264, 267, 292, 296,
                    342, 345,
                    # Regressor p5
                    258, 261, 297, 301,
                    261, 264, 302, 306,
                    264, 267, 307, 311,
                    342, 345,
                     # Regressor p6
                    258, 261, 312, 316,
                    261, 264, 317, 321,
                    264, 267, 322, 326,
                    342, 345,
                    # Regressor p7
                    258, 261, 327, 331,
                    261, 264, 332, 336,
                    264, 267, 337, 341,
                    342, 345,
                    # Classification
                    345, 348, 354, 358,
                    348, 351, 359, 363,
                    351, 354, 364, 368,
                    429, 432,
                    # Classification
                    345, 348, 369, 373,
                    348, 351, 374, 378,
                    351, 354, 379, 383,
                    429, 432,
                    # Classification
                    345, 348, 384, 388,
                    348, 351, 389, 393,
                    351, 354, 394, 398,
                    429, 432,
                    # Classification
                    345, 348, 399, 403,
                    348, 351, 404, 408,
                    351, 354, 409, 413,
                    429, 432,
                    # Classification
                    345, 348, 414, 418,
                    348, 351, 419, 423,
                    351, 354, 424, 428,
                    429, 432]

        for i_idx in range(int(len(index_l) // 2)):
            for idx in range(index_l[i_idx * 2], index_l[i_idx * 2 + 1]):  #
                key, w = weight_list[idx]
                if "num_batches_tracked" in key:
                    print(idx, "--------------------")
                    continue
                if len(w.shape) == 2:
                    print("transpose() \n")
                    w = w.transpose(1, 0)
                    w = w.cpu().data.numpy()
                else:
                    w = w.cpu().data.numpy()
                w.tofile(f)
                print(0, idx, key, w.shape)

# weigths extractor d5
if 0:  # weight download, (0 -> off, 1 -> on)
    weight_path = "E:/DEV4/mgmt/weights/eff-det/efficientdet-d5.weights"
    with open(weight_path, 'wb') as f:
        weights = model.state_dict()
        weight_list = [(key, value) for (key, value) in weights.items()]
        dumy = np.array([0] * 10, dtype=np.float32)
        dumy.tofile(f)

        for idx in range(776, len(weight_list)):  #
            key, w = weight_list[idx]
            if "num_batches_tracked" in key:
                print(idx, "--------------------")
                continue
            if len(w.shape) == 2:
                print("transpose() \n")
                w = w.transpose(1, 0)
                w = w.cpu().data.numpy()
            else:
                w = w.cpu().data.numpy()
            w.tofile(f)
            print(0, idx, key, w.shape)


        index_l = [
                    # Bifpn0
                    93, 99,
                    72, 92, #first time0 순서 바꿈
                    0, 1, 8, 15,
                    1, 2, 16, 23,
                    2, 3, 24, 31,
                    3, 4, 32, 39,
                    100, 113, #first time1
                    4, 5, 40, 47,
                    5, 6, 48, 55,
                    6, 7, 56, 63,
                    7, 8, 64, 71,
                    # Bifpn1
                    114, 115, 122, 129,
                    115, 116, 130, 137,
                    116, 117, 138, 145,
                    117, 118, 146, 153,
                    118, 119, 154, 161,
                    119, 120, 162, 169,
                    120, 121, 170, 177,
                    121, 122, 178, 185,
                    # Bifpn2
                    186, 187, 194, 201,
                    187, 188, 202, 209,
                    188, 189, 210, 217,
                    189, 190, 218, 225,
                    190, 191, 226, 233,
                    191, 192, 234, 241,
                    192, 193, 242, 249,
                    193, 194, 250, 257,
            # Bifpn3
            186+72, 187+72, 194+72, 201+72,
            187+72, 188+72, 202+72, 209+72,
            188+72, 189+72, 210+72, 217+72,
            189+72, 190+72, 218+72, 225+72,
            190+72, 191+72, 226+72, 233+72,
            191+72, 192+72, 234+72, 241+72,
            192+72, 193+72, 242+72, 249+72,
            193+72, 194+72, 250+72, 257+72,
            # Bifpn4
            186+72*2, 187+72*2, 194+72*2, 201+72*2,
            187+72*2, 188+72*2, 202+72*2, 209+72*2,
            188+72*2, 189+72*2, 210+72*2, 217+72*2,
            189+72*2, 190+72*2, 218+72*2, 225+72*2,
            190+72*2, 191+72*2, 226+72*2, 233+72*2,
            191+72*2, 192+72*2, 234+72*2, 241+72*2,
            192+72*2, 193+72*2, 242+72*2, 249+72*2,
            193+72*2, 194+72*2, 250+72*2, 257+72*2,
            # Bifpn5
            186+72*3, 187+72*3, 194+72*3, 201+72*3,
            187+72*3, 188+72*3, 202+72*3, 209+72*3,
            188+72*3, 189+72*3, 210+72*3, 217+72*3,
            189+72*3, 190+72*3, 218+72*3, 225+72*3,
            190+72*3, 191+72*3, 226+72*3, 233+72*3,
            191+72*3, 192+72*3, 234+72*3, 241+72*3,
            192+72*3, 193+72*3, 242+72*3, 249+72*3,
            193+72*3, 194+72*3, 250+72*3, 257+72*3,
            # Bifpn6
            186+72*4, 187+72*4, 194+72*4, 201+72*4,
            187+72*4, 188+72*4, 202+72*4, 209+72*4,
            188+72*4, 189+72*4, 210+72*4, 217+72*4,
            189+72*4, 190+72*4, 218+72*4, 225+72*4,
            190+72*4, 191+72*4, 226+72*4, 233+72*4,
            191+72*4, 192+72*4, 234+72*4, 241+72*4,
            192+72*4, 193+72*4, 242+72*4, 249+72*4,
            193+72*4, 194+72*4, 250+72*4, 257+72*4,


                    # Regressor p3
                    546, 549, 558, 562,
                    549, 552, 563, 567,
                    552, 555, 568, 572,
                    555, 558, 573, 577,
                    658, 661,

                    # Regressor p4
                    546, 549, 558+20, 562+20,
                    549, 552, 563+20, 567+20,
                    552, 555, 568+20, 572+20,
                    555, 558, 573+20, 577+20,
                    658, 661,
                    # Regressor p5
                    546, 549, 558+20*2, 562+20*2,
                    549, 552, 563+20*2, 567+20*2,
                    552, 555, 568+20*2, 572+20*2,
                    555, 558, 573+20*2, 577+20*2,
                    658, 661,
                     # Regressor p6
                    546, 549, 558+20*3, 562+20*3,
                    549, 552, 563+20*3, 567+20*3,
                    552, 555, 568+20*3, 572+20*3,
                    555, 558, 573+20*3, 577+20*3,
                    658, 661,
                    # Regressor p7
                    546, 549, 558+20*4, 562+20*4,
                    549, 552, 563+20*4, 567+20*4,
                    552, 555, 568+20*4, 572+20*4,
                    555, 558, 573+20*4, 577+20*4,
                    658, 661,

                    # Classification
                    661, 664, 673, 677,
                    664, 667, 678, 682,
                    667, 670, 683, 687,
                    670, 673, 688, 692,
                    773, 776,
                    # Classification
                    661, 664, 673+20, 677+20,
                    664, 667, 678+20, 682+20,
                    667, 670, 683+20, 687+20,
                    670, 673, 688+20, 692+20,
                    773, 776,
                    # Classification
                    661, 664, 673+20*2, 677+20*2,
                    664, 667, 678+20*2, 682+20*2,
                    667, 670, 683+20*2, 687+20*2,
                    670, 673, 688+20*2, 692+20*2,
                    773, 776,
                    # Classification
                    661, 664, 673+20*3, 677+20*3,
                    664, 667, 678+20*3, 682+20*3,
                    667, 670, 683+20*3, 687+20*3,
                    670, 673, 688+20*3, 692+20*3,
                    773, 776,
                    # Classification
                    661, 664, 673+20*4, 677+20*4,
                    664, 667, 678+20*4, 682+20*4,
                    667, 670, 683+20*4, 687+20*4,
                    670, 673, 688+20*4, 692+20*4,
                    773, 776]

        for i_idx in range(int(len(index_l) // 2)):
            for idx in range(index_l[i_idx * 2], index_l[i_idx * 2 + 1]):  #
                key, w = weight_list[idx]
                if "num_batches_tracked" in key:
                    print(idx, "--------------------")
                    continue
                if len(w.shape) == 2:
                    print("transpose() \n")
                    w = w.transpose(1, 0)
                    w = w.cpu().data.numpy()
                else:
                    w = w.cpu().data.numpy()
                w.tofile(f)
                print(0, idx, key, w.shape)


if use_cuda:
    model = model.cuda()
if use_float16:
    model = model.half()

# function for display
def display(preds, imgs):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            return imgs[i]

        for j in range(len(preds[i]['rois'])):
            score = float(preds[i]['scores'][j])
            (x1, y1, x2, y2) = preds[i]['rois'][j].astype(np.int)
            cv2.rectangle(imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
            obj = obj_list[preds[i]['class_ids'][j]]
            cv2.putText(imgs[i], '{}, {:.3f}'.format(obj, score),
                        (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 0), 1)
        
        return imgs[i]
# Box
regressBoxes = BBoxTransform()
clipBoxes = ClipBoxes()

# Video capture
cap = cv2.VideoCapture(video_src)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # frame preprocessing
    ori_imgs, framed_imgs, framed_metas = preprocess_video(frame, max_size=input_size)

    if use_cuda:
        x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
    else:
        x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

    x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

    # model predict
    t1 = time.time()
    with torch.no_grad():

        features, regression, classification, anchors = model(x)

        out = postprocess(x,
                        anchors, regression, classification,
                        regressBoxes, clipBoxes,
                        threshold, iou_threshold)

    # result
    out = invert_affine(framed_metas, out)

    tact_time = (time.time() - t1)
    text = "fps::{: .2f}".format(1 / (tact_time))
    #print(f'{tact_time} seconds, {1 / tact_time} FPS, @batch_size 1')

    img_show = display(out, ori_imgs)

    cv2.putText(img_show, text, (int(5), int(img_show.shape[0] - 10)), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

    # show frame by frame
    cv2.imshow('frame',img_show)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()





