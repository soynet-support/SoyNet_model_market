import torch
import numpy as np
# batch_norm이 있지만 detect.py에서 사용할 때 퓨전해서 모델 가져옴

model_type = "s6" # n, s, m, l, x
                # s6, m6, l6, x6

model = torch.load("./weights/yolov5%s.pt"%(model_type))["model"]

if 0:
    weights = model.state_dict()
    weight_list = [(key, value) for (key, value) in weights.items()]
    newfile = open('./weights/weight_structure_%s.txt'%(model_type), 'w', encoding='utf-8')
    for idx in range(len(weight_list)):  #
        key, w = weight_list[idx]
        if "num_batches_tracked" in key:
            print(idx, "--------------------")
            newfile.write(f"{idx}--------------------\n")
            continue
        if len(w.shape) == 2:
            print("transpose() \n")
            newfile.write(f"transpose() \n")
            w = w.transpose(1, 0)
            w = w.cpu().data.numpy()
        else:
            w = w.cpu().data.numpy()
        print(0, idx, key, w.shape)
        newfile.write(f"0 {idx} {key} {w.shape}\n")
    newfile.close()
    exit(-1)

# yolov5, (0 -> off, 1 -> on)
if 1:
    weight_path = "./weights/yolov5%s.weights"%(model_type)  # 생성할 소이넷 웨이트 파일
    with open(weight_path, 'wb') as f:
        weights = model.state_dict()
        weight_list = [(key, value) for (key, value) in weights.items()]
        dumy = np.array([0] * 10, dtype=np.float32)
        dumy.tofile(f)

        anchors = 3

        if(model_type == "n" or model_type == "s"):
            index = [0, 18, 30, 42, 18, 30,
                     42, 54, 66, 90, 54, 66,
                     90, 102, 114, 150, 102, 114,
                     150, 162, 174, 186, 162, 174,
                     186, 210, 222, 234, 210, 222, #9
                     234, 246, 258, 270, 246, 258,
                     270, 282, 294, 306, 282, 294,
                     306, 318, 330, 342, 318, 330,
                     343, 349,
                     342, 343    # anchor
                     ]

        if (model_type == "m"):
            index = [0, 18, 30, 54, 18, 30,
                     54, 66, 78, 126, 66, 78,
                     126, 138, 150, 222, 138, 150,
                     222, 234, 246, 270, 234, 246,
                     270, 294, 306, 330, 294, 306, #9
                     330, 342, 354, 378, 342, 354,
                     378, 390, 402, 426, 390, 402,
                     426, 438, 450, 474, 438, 450,
                     475, 481,
                     474, 475   # anchor
                     ]

        if (model_type == "l"):
            index = [0, 18, 30, 66, 18, 30,
                     66, 78, 90, 162, 78, 90,
                     162, 174, 186, 294, 174, 186,
                     294, 306, 318, 354, 306, 318,
                     354, 378, 390, 426, 378, 390, #9
                     426, 438, 450, 486, 438, 450,
                     486, 498, 510, 546, 498, 510,
                     546, 558, 570, 606, 558, 570,
                     607, 613,
                     606, 607 # anchor
                     ]

        if(model_type == "x"):
            index = [0, 18, 30, 78, 18, 30,
                     78, 90, 102, 198, 90, 102,
                     198, 210, 222, 366, 210, 222,
                     366, 378, 390, 438, 378, 390,
                     438, 462, 474, 522, 462, 474, #9
                     522, 534, 546, 594, 534, 546,
                     594, 606, 618, 666, 606, 618,
                     666, 678, 690, 738, 678, 690,
                     739, 745,
                     738, 739   # anchor
                     ]

        # P6
        if (model_type == "s6" or model_type == "n6"):
            index = [0, 18, 30, 42, 18, 30,
                     42, 54, 66, 90, 54, 66,
                     90, 102, 114, 150, 102, 114,
                     150, 162, 174, 186, 162, 174,
                     186, 198, 210, 222, 198, 210,
                     222, 246, 258, 270, 246, 258,
                     270, 282, 294, 306, 282, 294,
                     306, 318, 330, 342, 318, 330,
                     342, 354, 366, 378, 354, 366,
                     378, 390, 402, 414, 390, 402,
                     414, 426, 438, 450, 426, 438,
                     451, 459,
                     450, 451 # anchor
                     ]

            anchors = 4

        if (model_type == "m6"):
            index = [0, 18, 30, 54, 18, 30,
                     54, 66, 78, 126, 66, 78,
                     126, 138, 150, 222, 138, 150,
                     222, 234, 246, 270, 234, 246,
                     270, 282, 294, 318, 282, 294,
                     318, 342, 354, 378, 342, 354,
                     378, 390, 402, 426, 390, 402,
                     426, 438, 450, 474, 438, 450,
                     474, 486, 498, 522, 486, 498,
                     522, 534, 546, 570, 534, 546,
                     570, 582, 594, 618, 582, 594,
                     619, 627,
                     618, 619# anchor
                     ]

            anchors = 4

        if (model_type == "l6"):
            index = [0, 18, 30, 66, 18, 30,
                     66, 78, 90, 162, 78, 90,
                     162, 174, 186, 294, 174, 186,
                     294, 306, 318, 354, 306, 318,
                     354, 366, 378, 414, 366, 378,
                     414, 438, 450, 486, 438, 450,
                     486, 498, 510, 546, 498, 510,
                     546, 558, 570, 606, 558, 570,
                     606, 618, 630, 666, 618, 630,
                     666, 678, 690, 726, 678, 690,
                     726, 738, 750, 786, 738, 750,
                     787, 795,
                     786, 787  # anchor
                     ]

            anchors = 4

        if (model_type == "x6"):
            index = [0, 18, 30, 78, 18, 30,
                     78, 90, 102, 198, 90, 102,
                     198, 210, 222, 366, 210, 222,
                     366, 378, 390, 438, 378, 390,
                     438, 450, 462, 510, 450, 462,
                     510, 534, 546, 594, 534, 546,
                     594, 606, 618, 666, 606, 618,
                     666, 678, 690, 738, 678, 690,
                     738, 750, 762, 810, 750, 762,
                     810, 822, 834, 882, 822, 834,
                     882, 894, 906, 954, 894, 906,
                     955, 963,
                     954, 955 # anchor
                     ]

            anchors = 4

        anchor_layer_idx = index[-2]

        for i_idx in range(int(len(index) / 2)):
            for idx in range(index[i_idx * 2], index[i_idx * 2 + 1]):  #
                key, w = weight_list[idx]
                if (idx == anchor_layer_idx): # anchor_grid
                    for i in range(anchors):
                        w_ = w[i] * (8 * (2 ** i))  # stride : 8, 16, 32, 64
                        w_ = w_.cpu().data.numpy().astype(np.float32)
                        w_.tofile(f)
                    print(0, idx, key, w.shape)
                    continue

                if "num_batches_tracked" in key:
                    print(idx, "--------------------")
                    continue
                if len(w.shape) == 2:
                    print("transpose() \n")
                    w = w.transpose(1, 0)
                    w = w.cpu().data.numpy().astype(np.float32)
                else:
                    w = w.cpu().data.numpy().astype(np.float32)
                w.tofile(f)
                print(0, idx, key, w.shape)
    f.close()