import argparse
import numpy as np
import torch
from model import IDN

parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, default='IDN')
parser.add_argument('--weights_path', type=str, default="./weights/IDN_epoch_19.pth")
parser.add_argument('--scale', type=int, default=2)
parser.add_argument('--num_features', type=int, default=64)
parser.add_argument('--d', type=int, default=16)
parser.add_argument('--s', type=int, default=4)
opt = parser.parse_args()

# model = IDN(opt)

model = torch.load(opt.weights_path)

if 1:
    with open("IDN_pytorch.weights",
              'wb') as f:
        weight_list = [(key, value) for (key, value) in model.items()]
        dumy = np.array([0] * 10, dtype=np.float32)
        dumy.tofile(f)
        for idx in range(0, len(weight_list)):
            key, w = weight_list[idx]
            if "num_batches_tracked" in key:
                print(idx, "--------------------")
                continue
            if len(w.shape) == 2:
                w = w.transpose(1, 0)
                w = w.cpu().data.numpy()
            else:
                w = w.cpu().data.numpy()
            w.tofile(f)
            print(0, idx, key, w.shape)
    print('웨이트 생성 완료')
