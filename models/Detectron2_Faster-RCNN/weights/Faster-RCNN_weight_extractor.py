import pickle
import numpy as np
from fvcore.common.file_io import PathManager

# https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md
# COCO Object Detection Baselines - Faster R-CNN:R101-FPN
filename= "model_final_f6e8b1.pkl"
with PathManager.open(filename, "rb") as f :
    data = pickle.load(f, encoding="latin1")

#for idx, (key,value) in enumerate(data['model'].items()):
    #print(idx, key, value.shape)

wightfile = "fasterrcnn-101-detectron2.weight"
#with open(wightfile, 'wb') as f :
#    weight_list = [(key, value) for (key, value) in data['model'].items()]
    #for idx in range(len(weight_list)) :
    #    key, w = weight_list[idx]
    #    print(0, idx, key, w.shape)

with open(wightfile, 'wb') as f:
    weight_list = [(key, value) for (key, value) in data['model'].items()]
    dumy = np.array([0] * 10, dtype=np.float32)
    dumy.tofile(f)


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
            w = np.transpose(w, (1, 0))
        w.tofile(f)
        print(0, idx, key, w.shape)



