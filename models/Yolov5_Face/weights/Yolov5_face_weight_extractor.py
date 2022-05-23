# -*- coding: UTF-8 -*-
import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import copy

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    #clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords

def show_results(img, xywh, conf, landmarks, class_num):
    h,w,c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
    y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
    x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
    y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
    cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)

    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]

    for i in range(5):
        point_x = int(landmarks[2 * i] * w)
        point_y = int(landmarks[2 * i + 1] * h)
        cv2.circle(img, (point_x, point_y), tl+1, clors[i], -1)

    tf = max(tl - 1, 1)  # font thickness
    label = str(conf)[:5]
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img



def detect_one(model, image_path, img_size, device):
    # Load model
    conf_thres = 0.3
    iou_thres = 0.5

    orgimg = cv2.imread(image_path)  # BGR

    img0 = copy.deepcopy(orgimg)
    assert orgimg is not None, 'Image Not Found ' + image_path
    h0, w0 = orgimg.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)),0,0, interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

    img = letterbox(img0, new_shape=imgsz)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

    # Run inference
    t0 = time.time()

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression_face(pred, conf_thres, iou_thres)

    print('img.shape: ', img.shape)
    print('orgimg.shape: ', orgimg.shape)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        gn = torch.tensor(orgimg.shape)[[1, 0, 1, 0]].to(device)  # normalization gain whwh
        gn_lks = torch.tensor(orgimg.shape)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]].to(device)  # normalization gain landmarks
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], orgimg.shape).round()

            for j in range(det.size()[0]):
                xywh = (xyxy2xywh(det[j, :4].view(1, 4)) / gn).view(-1).tolist()
                conf = det[j, 4].cpu().numpy()
                landmarks = (det[j, 5:15].view(1, 10) / gn_lks).view(-1).tolist()
                class_num = det[j, 15].cpu().numpy()
                orgimg = show_results(orgimg, xywh, conf, landmarks, class_num)

    cv2.imwrite('result.jpg', orgimg)
    cv2.imshow('test', orgimg)
    cv2.waitKey(0)


def detect_cam(model, video_path,  img_size,device, thres=0.8):

    vs = cv2.VideoCapture(video_path)  # video
    conf_thres = thres
    iou_thres = 0.5

    while (vs.isOpened()):
        _, orgimg = vs.read()

        begin_time = time.time()
        img0 = copy.deepcopy(orgimg)
        assert orgimg is not None, 'Image Not Found ' + video_path
        h0, w0 = orgimg.shape[:2]  # orig hw
        r = img_size / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

        imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

        img = letterbox(img0, new_shape=imgsz)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

        # Run inference
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference

        pred = model(img)[0]

        # Apply NMS
        pred = non_max_suppression_face(pred, conf_thres, iou_thres)

        #print('img.shape: ', img.shape)
        #print('orgimg.shape: ', orgimg.shape)
        end_time = time.time()
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            gn = torch.tensor(orgimg.shape)[[1, 0, 1, 0]].to(device)  # normalization gain whwh
            gn_lks = torch.tensor(orgimg.shape)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]].to(device)  # normalization gain landmarks

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], orgimg.shape).round()

                for j in range(det.size()[0]):
                    conf = det[j, 4].cpu().numpy()
                    #if conf < thres :
                    #    continue
                    xywh = (xyxy2xywh(det[j, :4].view(1, 4)) / gn).view(-1).tolist()
                    landmarks = (det[j, 5:15].view(1, 10) / gn_lks).view(-1).tolist()
                    class_num = det[j, 15].cpu().numpy()
                    orgimg = show_results(orgimg, xywh, conf, landmarks, class_num)


        dur_time = end_time - begin_time
        fps = 1 / (dur_time + 1e-5)
        text = "fps=%.2f " % (fps)
        print(text)
        cv2.putText(orgimg, text, (5, h0 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

        #cv2.imwrite('result.jpg', orgimg)
        cv2.imshow('test', orgimg)
        key = cv2.waitKey(1)
        if key == 27 or key == 1048603:  # ESC
            break

import asyncio
import websockets
import cv2
import base64

async def detect_cam2(model, video_path,  img_size,device, thres=0.8):
    async with websockets.connect("ws://localhost:8000/flag=python") as websocket:

        vs = cv2.VideoCapture(video_path)  # video
        conf_thres = thres
        iou_thres = 0.5

        while (vs.isOpened()):
            _, orgimg = vs.read()

            begin_time = time.time()
            img0 = copy.deepcopy(orgimg)
            assert orgimg is not None, 'Image Not Found ' + video_path
            h0, w0 = orgimg.shape[:2]  # orig hw
            r = img_size / max(h0, w0)  # resize image to img_size
            if r != 1:  # always resize down, only resize up if training with augmentation
                interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
                img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

            imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

            img = letterbox(img0, new_shape=imgsz)[0]
            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

            # Run inference
            img = torch.from_numpy(img).to(device)
            img = img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference

            pred = model(img)[0]

            # Apply NMS
            pred = non_max_suppression_face(pred, conf_thres, iou_thres)

            #print('img.shape: ', img.shape)
            #print('orgimg.shape: ', orgimg.shape)
            end_time = time.time()
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                gn = torch.tensor(orgimg.shape)[[1, 0, 1, 0]].to(device)  # normalization gain whwh
                gn_lks = torch.tensor(orgimg.shape)[[1, 0, 1, 0, 1, 0, 1, 0, 1, 0]].to(device)  # normalization gain landmarks

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class

                    det[:, 5:15] = scale_coords_landmarks(img.shape[2:], det[:, 5:15], orgimg.shape).round()

                    for j in range(det.size()[0]):
                        conf = det[j, 4].cpu().numpy()
                        #if conf < thres :
                        #    continue
                        xywh = (xyxy2xywh(det[j, :4].view(1, 4)) / gn).view(-1).tolist()
                        landmarks = (det[j, 5:15].view(1, 10) / gn_lks).view(-1).tolist()
                        class_num = det[j, 15].cpu().numpy()
                        orgimg = show_results(orgimg, xywh, conf, landmarks, class_num)


            dur_time = end_time - begin_time
            fps = 1 / (dur_time + 1e-5)
            text = "fps=%.2f " % (fps)
            print(text)
            cv2.putText(orgimg, text, (5, h0 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

            #cv2.imwrite('result.jpg', orgimg)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            result, encimg = cv2.imencode('.jpg', orgimg, encode_param)
            encoded = base64.b64encode(encimg).decode('utf-8')
            await websocket.send(encoded)
            #cv2.imshow('test', orgimg)
            #key = cv2.waitKey(1)
            #if key == 27 or key == 1048603:  # ESC
            #    break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--weights', nargs='+', type=str, default='./yolov5n-0.5.pt', help='model.pt path(s)')
    #parser.add_argument('--weights', nargs='+', type=str, default='./yolov5n-face.pt',help='model.pt path(s)')
    parser.add_argument('--weights', nargs='+', type=str, default='./yolov5s-face.pt',help='model.pt path(s)')
    parser.add_argument('--image', type=str, default='E:/DEV4/data/zidane.jpg', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--video', type=str, default='data/images/BTS.mp4', help='source')  # file/folder, 0 for webcam
    #parser.add_argument('--video', type=str, default=0, help='source')  # file/folder, 0 for webcam

    parser.add_argument('--imgsize', type=int, default=640, help='inference size (pixels)')
    opt = parser.parse_args()
    print(opt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(opt.weights, device)

    if 0:
        import numpy as np
        weight_path = "E:/DEV4/mgmt/weights/yolov5sn-face.weights"
        #weight_path = "yolov5sn.weights"
        with open(weight_path, 'wb') as f:
            weights = model.state_dict()
            weight_list = [(key, value) for (key, value) in weights.items()]
            dumy = np.array([0] * 10, dtype=np.float32)
            dumy.tofile(f)
            index_l = [0, 6, 10, 14, 6, 10, 14, 18, 22, 34, 18, 22, 34, 38, 42, 54,
                       38, 42, 54, 62, 66, 70, 62, 66, 70, 74, 78, 82, 74, 78, 82, 86,
                       90, 94, 86, 90, 94, 98, 102, 106, 98, 102, 106, 110, 114, 118,
                       110, 114, 120, 126, 119, 120]

            for i_idx in range(int(len(index_l) / 2)):
                temp = 4
                if i_idx == 0 :
                    temp = 0
                for idx in range(index_l[i_idx * 2]+temp, index_l[i_idx * 2 + 1]+4):  #
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

    if 0:
        weights = model.state_dict()
        weight_list = [(key, value) for (key, value) in weights.items()]
        for idx in range(len(weight_list)):  #
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
    #detect_one(model, opt.image, opt.imgsize, device)
    detect_cam(model, opt.video, opt.imgsize, device)
    #asyncio.run(detect_cam2(model, 0, opt.imgsize, device))
