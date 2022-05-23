import pickle
import numpy as np
from fvcore.common.file_io import PathManager

# https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md
# COCO Instance Segmentation

filename= "model_final_a3ec72.pkl"                  ### Mask R-CNN: R101-FPN
# filename= "mask_rcnn_X_101_32x8d_FPN_3x.pkl"      ### Mask R-CNN: X_101_32x8d_FPN_3x

with PathManager.open(filename, "rb") as f :
    data = pickle.load(f, encoding="latin1")


backbone = "r101" # x101

wightfile = "maskrcnn-{}-fpn-detectron2.weights".format(backbone)

with open(wightfile, 'wb') as f :
    weight_list = [(key, value) for (key, value) in data['model'].items()]
    dumy = np.array([0] * 10, dtype=np.float32)
    dumy.tofile(f)

    if 0:   # weight_list show all
        for i in range(len(weight_list)) :
            key, w = weight_list[i]
            print(0, i, key, w.shape)
        exit()

    ########
    # r101
    ########
    if backbone == "r101":
        if (len(weight_list) == 567):
            for idx in range(16, 536):  # resnet
                key, w = weight_list[idx]
                w.tofile(f)
                print(0, idx, key, w.shape)

            for idx in range(12, -1, -4):  # fpn
                for idx_2 in range(4) :
                    key, w = weight_list[idx+idx_2]
                    w.tofile(f)
                    print(0, idx+idx_2, key, w.shape)

            for r_idx in range(5): # rpn * 5 반복
                for idx in range(541, 547):
                    key, w = weight_list[idx]
                    w.tofile(f)
                    print(0, idx, key, w.shape)

            for idx in range(547, 555):  # box head
                key, w = weight_list[idx]
                if len(w.shape) == 2 :
                    w = np.transpose(w, (1,0))
                w.tofile(f)
                print(0, idx, key, w.shape)

            for idx in range(555, 567):  # mask_head
                key, w = weight_list[idx]
                w.tofile(f)
                print(0, idx, key, w.shape)

        else:
            for idx in range(16, 536):  # resnet
                key, w = weight_list[idx]
                w.cpu().numpy().tofile(f)
                print(0, idx, key, w.shape)

            for idx in range(12, -1, -4):  # fpn
                for idx_2 in range(4):
                    key, w = weight_list[idx + idx_2]
                    w.cpu().numpy().tofile(f)
                    print(0, idx + idx_2, key, w.shape)

            for r_idx in range(5):  # rpn * 5 반복
                for idx in range(541, 547):
                    key, w = weight_list[idx]
                    w.cpu().numpy().tofile(f)
                    print(0, idx, key, w.shape)

            for idx in range(547, 555):  # box head
                key, w = weight_list[idx]
                w = w.cpu().numpy()

                if len(w.shape) == 2:
                    w = np.transpose(w, (1, 0))
                w.tofile(f)
                print(0, idx, key, w.shape)

    ########
    # x101
    ########
    if backbone == "x101":
        if len(weight_list) < 567:  # 버리는 값이 없어 짧음
            for idx in range(16, 536):  # resnet
                key, w = weight_list[idx]
                w.cpu().numpy().tofile(f)
                print(0, idx, key, w.shape)

            for idx in range(12, -1, -4):  # fpn
                for idx_2 in range(4):
                    key, w = weight_list[idx + idx_2]
                    w.cpu().numpy().tofile(f)
                    print(0, idx + idx_2, key, w.shape)

            for r_idx in range(5):  # rpn * 5 반복
                for idx in range(541, 547):
                    key, w = weight_list[idx]
                    w.cpu().numpy().tofile(f)
                    print(0, idx, key, w.shape)

            for idx in range(547, 555):  # box head
                key, w = weight_list[idx]
                w = w.cpu().numpy()

                if len(w.shape) == 2:
                    w = np.transpose(w, (1, 0))
                w.tofile(f)
                print(0, idx, key, w.shape)

        else:
            for idx in range(16, 536):  # resnet
                key, w = weight_list[idx]
                w.tofile(f)
                print(0, idx, key, w.shape)

            for idx in range(12, -1, -4):  # fpn
                for idx_2 in range(4):
                    key, w = weight_list[idx + idx_2]
                    w.tofile(f)
                    print(0, idx + idx_2, key, w.shape)

            for r_idx in range(5):  # rpn * 5 반복
                for idx in range(541, 547):
                    key, w = weight_list[idx]
                    w.tofile(f)
                    print(0, idx, key, w.shape)

            for idx in range(547, 555):  # box head
                key, w = weight_list[idx]
                if len(w.shape) == 2:
                    w = np.transpose(w, (1, 0))
                w.tofile(f)
                print(0, idx, key, w.shape)

            for idx in range(555, 567):  # mask_head
                key, w = weight_list[idx]
                w.tofile(f)
                print(0, idx, key, w.shape)

