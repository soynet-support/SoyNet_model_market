import numpy as np
import cv2 as cv
from utils.ClassName import COCO_80, COCO_90


def MakeMultiple32(value):
    remainder = value % 32
    print(remainder)
    if remainder != 0:
        if remainder < 16:
            return value - remainder
        else:
            return value + 32 - remainder
    else:
        return value


def MaskedLM_Data(value):
    data_length = len(value)

    # input_ids
    input_data = value

    # position_ids
    for idx in range(data_length):
        input_data = np.append(input_data, [idx])

    # token_type_ids
    for idx in range(data_length):
        input_data = np.append(input_data, [0])

    return input_data


def CreateNetsizeImage(img, netw, neth, scale):
    newh = neth
    s = newh / img.shape[0]
    neww = img.shape[1] * s
    if neww > netw:
        neww = netw
        s = neww / img.shape[1]
        newh = img.shape[0] * s
    scale = 1 / s
    return int(neww), int(newh)


def ViewResult(ori_img, output, name, batch=1, nms=0):

    if ori_img is None:
        exit()

    row = ori_img.shape[0]
    col = ori_img.shape[1]

    if name == 'IDN':
        output = cv.cvtColor(output, cv.COLOR_BGR2RGB)
        cv.imshow(name + ' Image', output)
        cv.waitKey(0)

    elif name == 'Glean':
        output = np.reshape(output, (output.shape[2], output.shape[3], output.shape[1]))
        output = cv.cvtColor(output, cv.COLOR_BGR2RGB)
        cv.imshow(name + ' Image', output)
        cv.waitKey(0)

    elif name == 'Pix2Pix' or name == 'FAnoGan':
        output = output.reshape((output.shape[1], output.shape[2], output.shape[3]))
        output = np.transpose(output, (1, 2, 0))
        cv.imshow(name + ' Image', output)
        cv.waitKey(0)

    elif name == 'CycleGan':
        img = np.reshape(output, (row, col, 3)).astype(np.uint8)
        cv.imshow(name + ' Image', img)
        cv.waitKey(0)

    elif name == 'Faster-RCNN' or name == 'EfficientDet' or name == 'Yolor' or name == 'Yolov5':
        class_name = COCO_80()
        for b_idx in range(batch):
            for n_idx in range(nms):
                x1, y1, x2, y2, obj_id, prob = output[n_idx + b_idx * nms]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv.rectangle(ori_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv.putText(ori_img, class_name[obj_id], (x1, y1 - 3), 1, 1.5, (255, 0, 0), 1, cv.LINE_AA)
            cv.imshow(name + ' Image', ori_img)
            cv.waitKey(0)

    elif name == 'SSD_Mobilenet' or name == 'Yolov3' or name == 'Yolov4':
        class_name = COCO_90()
        for b_idx in range(batch):
            for n_idx in range(nms):
                x1, y1, x2, y2, obj_id, prob = output[n_idx + b_idx * nms]
                x1, x2 = int(x1 * ori_img.shape[1]), int(x2 * ori_img.shape[1])
                y1, y2 = int(y1 * ori_img.shape[0]), int(y2 * ori_img.shape[0])
                cv.rectangle(ori_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv.putText(ori_img, class_name[obj_id], (int(x1), int(y1) - 3), 1, 1.5, (255, 0, 0), 1, cv.LINE_AA)
            cv.imshow(name + ' Image', ori_img)
            cv.waitKey(0)

    elif name == 'Mask-RCNN':
        class_name = COCO_80()
        for b_idx in range(batch):
            rip, masked_img = output[b_idx]
            for r_idx in rip:
                x1, y1, x2, y2, obj_id, prob = r_idx
                cv.rectangle(masked_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv.putText(masked_img, class_name[obj_id], (int(x1), int(y1) - 3), 1, 1.5, (255, 0, 0), 1, cv.LINE_AA)
            cv.imshow(name + ' Image', masked_img)
            cv.waitKey(0)

    elif name == 'Yolact++' or name == 'Yolact':
        class_name = COCO_80()
        for b_idx in range(batch):
            rip, masked_img = output[b_idx]
            for r_idx in rip:
                x1, y1, x2, y2, obj_id, prob = r_idx
                x1, x2 = int(x1 * ori_img.shape[1]), int(x2 * ori_img.shape[1])
                y1, y2 = int(y1 * ori_img.shape[0]), int(y2 * ori_img.shape[0])
                cv.rectangle(masked_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv.putText(masked_img, class_name[obj_id], (x1, y1 - 3), 1, 1.5, (255, 0, 0), 1, cv.LINE_AA)
            cv.imshow(name + ' Image', masked_img)
            cv.waitKey(0)

    elif name == 'Pose-RCNN':
        for b_idx in range(batch):
            result = output[b_idx]
            rip = result[: nms * 6]
            keypoint = result[nms * 6:]
            for n_idx in range(nms):
                x1, y1, x2, y2, obj_id, prob = rip[n_idx * 6: (n_idx + 1) * 6]  # 6 + 17 * 3
                cv.rectangle(ori_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                for k_idx in range(n_idx * 51, (n_idx + 1) * 51, 3):
                    cv.circle(ori_img, (keypoint[k_idx], keypoint[k_idx + 1]), 5, (0, 255, 255), -1)
            cv.imshow(name + ' Image', ori_img)
            cv.waitKey(0)

    elif name == 'RetinaFace' or name == 'Yolov5_face':
        for n_idx in range(nms):
            result = output[n_idx]
            cv.rectangle(ori_img, (result[0], result[1]), (result[2], result[3]), (0, 0, 255), 2)
            for idx in range(5, 15, 2):
                cv.circle(ori_img, (result[idx], result[idx + 1]), 5, (0, 255, 255), -1)
        cv.imshow(name + ' Image', ori_img)
        cv.waitKey(0)
