import _init_paths

import numpy as np
import re
import torch
from opts import opts
from tracker.multitracker import JDETracker

opt = opts().init()

opt.load_model = "../models/fairmot_dla34.pth"
opt.conf_thres = 0.4
tracker = JDETracker(opt, frame_rate=15)
model = tracker.model

weight_path = "fairmot_dcn_v2.weights"
weights = model.state_dict()
weight_list = [(key, value) for (key, value) in weights.items()]

if 1:
    with open("fairmot_dcn_v2_weight_keys.txt", "w") as f:
        for idx in range(len(weight_list)):  #
            key, w = weight_list[idx]
            if "num_batches_tracked" in key:
                print(idx, "--------------------")
                continue
            if len(w.shape) == 2:
                # print("transpose() ")
                w = w.transpose(1, 0)
                w = w.cpu().data.numpy()
            else:
                w = w.cpu().data.numpy()
            line = "0 {} {} {}".format(idx, key, w.shape)
            print(0, idx, key, w.shape)
            f.write(line + "\n")
    f.close()

if 1:
    with open(weight_path, 'wb') as f:
        dumy = np.array([0] * 10, dtype=np.float32)
        dumy.tofile(f)

        for idx in range(0, len(weight_list)):  # box head
            key, w = weight_list[idx]
            if any(skip in key for skip in ('num_batches_tracked', 'base.fc')): # no use
                print(idx, "--------------------")
                continue

            if any(skip in key for skip in ('project', 'conv_offset_mask')):
                continue

            # early use DeformConv
            df = re.compile('["proj_"|"node_"][0-9].conv.["weight"|"bias"]')
            mc = df.findall(key)
            if mc:
                continue

            if "tree1.conv1" in key:
                # project first
                for i in range(5): # wrbmv
                    idx_= idx + 30 + i
                    key_p, w_p = weight_list[idx_]
                    w_p = w_p.cpu().data.numpy()
                    w_p.tofile(f)
                    print(0, idx_, key_p, w_p.shape)
                print("")

            if "actf.0.weight" in key:
                # Conv first
                for i in range(2):  # wa
                    idx_ = idx + 7 + i
                    key_d, w_d = weight_list[idx_]
                    w_d = w_d.cpu().data.numpy()
                    w_d.tofile(f)
                    print(0, idx_, key_d, w_d.shape)
                # DefromConv first
                for i in range(2):  # wa
                    idx_ = idx + 5 + i
                    key_d, w_d = weight_list[idx_]
                    w_d = w_d.cpu().data.numpy()
                    w_d.tofile(f)
                    print(0, idx_, key_d, w_d.shape)

                print("")

            if len(w.shape) == 2:
                w = w.transpose(1, 0)
                w = w.cpu().data.numpy()
            else:
                w = w.cpu().data.numpy()
            w.tofile(f)
            print(0, idx, key, w.shape)
    print('웨이트 생성 완료')
