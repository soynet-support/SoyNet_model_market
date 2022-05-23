
from __future__ import print_function
import os
import cv2
from models1 import *
import torch
import numpy as np
import time, sys, pickle
from config1.config import Config
from torch.nn import DataParallel


def get_lfw_list(pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    data_list = []
    for pair in pairs:
        splits = pair.split()

        if splits[0] not in data_list:
            data_list.append(splits[0])

        if splits[1] not in data_list:
            data_list.append(splits[1])
    return data_list


def get_s_list(s_list):
    with open(s_list, 'r') as fd:
        pairs = fd.readlines()
    data_list = []
    for pair in pairs:
        splits = pair.split()

        if splits[0] not in data_list:
            data_list.append(splits[0])

        # if splits[1] not in data_list:
        #    data_list.append(splits[1])
    return data_list


def load_image(img_path):
    image = cv2.imread(img_path, 0)
    if image is None:
        return None
    image = np.dstack((image, np.fliplr(image)))
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    return image


def get_featurs(model, test_list, batch_size=10):
    images = None
    features = None
    cnt = 0
    for i, img_path in enumerate(test_list):
        image = load_image(str(img_path))
        if image is None:
            print('read {} error'.format(img_path))

        if images is None:
            images = image
        else:
            images = np.concatenate((images, image), axis=0)

        if images.shape[0] % batch_size == 0 or i == len(test_list) - 1:
            cnt += 1

            data = torch.from_numpy(images)
            data = data.to(torch.device("cuda"))
            output = model(data)
            output = output.data.cpu().numpy()

            fe_1 = output[::2]
            fe_2 = output[1::2]
            feature = np.hstack((fe_1, fe_2))
            # print(feature.shape)

            if features is None:
                features = feature
            else:
                features = np.vstack((features, feature))

            images = None

    return features, cnt


def get_feature(model, test):
    images = None
    features = None

    iters = 1
    for _ in range(iters):
        start = time.time()
        image = load_image(str(test))
        data = torch.from_numpy(image)
        data = data.to(torch.device("cuda"))
        output = model(data)
        end = time.time()
        print("fps = ", 1. / (end - start))

    output = output.data.cpu().numpy()

    fe_1 = output[::2]
    fe_2 = output[1::2]
    feature = np.hstack((fe_1, fe_2))
    # print(feature.shape)

    return feature


def load_model(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def get_feature_dict(test_list, features):
    fe_dict = {}
    for i, each in enumerate(test_list):
        # key = each.split('/')[1]
        fe_dict[each] = features[i]
    return fe_dict


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def cal_accuracy(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th

    return (best_acc, best_th)


def test_performance(fe_dict, pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()

    sims = []
    labels = []
    for pair in pairs:
        splits = pair.split()
        fe_1 = fe_dict[splits[0]]
        fe_2 = fe_dict[splits[1]]
        label = int(splits[2])
        sim = cosin_metric(fe_1, fe_2)

        sims.append(sim)
        labels.append(label)

    acc, th = cal_accuracy(sims, labels)
    return acc, th


def lfw_test(model, img_paths, identity_list, compair_list, batch_size):
    s = time.time()
    features, cnt = get_featurs(model, img_paths, batch_size=batch_size)
    print(features.shape)
    t = time.time() - s
    print('total time is {}, average time is {}'.format(t, t / cnt))
    fe_dict = get_feature_dict(identity_list, features)
    acc, th = test_performance(fe_dict, compair_list)
    print('lfw face verification accuracy: ', acc, 'threshold: ', th)
    return acc


if __name__ == '__main__':

    # 모델 선택
    model = resnet_face18(False)

    # 모델 불러오기
    model = DataParallel(model)
    model.load_state_dict(torch.load('./checkpoints/resnet18_110.pth'))
    model.to(torch.device("cuda"))
    model.eval()

    weight_path = "arc_face_r18.weights"

    img_path = '37_128x128.jpg'
    # # 식별할 이미지의 feature 뽑기
    feature = get_feature(model, img_path)  # feature 계산
    print(feature.shape)

    # weigths extractor

    if 0:  # weight download, (0 -> off, 1 -> on)
        print()
        with open(weight_path, 'wb') as f:
            weights = model.state_dict()
            weight_list = [(key, value) for (key, value) in weights.items()]

            f.write(np.array([0] * 10, dtype=np.float32)) # dummy 10 line

            if 0:  # 전체 보기
                for idx in range(len(weight_list)):
                    key, value = weight_list[idx]
                    if "num_batches_tracked" in key:
                        print(idx, "--------------------")
                        continue
                    print(idx, key, value.shape)
                exit()

            if 1:
                for idx in range(0, 24):  # BACKBONE(resnet18)
                    key, value = weight_list[idx]
                    if "num_batches_tracked" in key:
                        print(idx, "--------------------")
                        continue
                    w = value.cpu().data.numpy()
                    f.write(w)
                    print(0, idx, key, value.shape)
                print()

                for idx in range(18, 19):  # prelu layer1
                    key, value = weight_list[idx]
                    if "num_batches_tracked" in key:
                        print(idx, "--------------------")
                        continue
                    w = value.cpu().data.numpy()
                    f.write(w)
                    print(0, idx, key, value.shape)
                print()

                for idx in range(24, 42):  # BACKBONE(resnet18)
                    key, value = weight_list[idx]
                    if "num_batches_tracked" in key:
                        print(idx, "--------------------")
                        continue
                    w = value.cpu().data.numpy()
                    f.write(w)
                    print(0, idx, key, value.shape)
                print()

                for idx in range(36, 37):  # prelu layer1
                    key, value = weight_list[idx]
                    if "num_batches_tracked" in key:
                        print(idx, "--------------------")
                        continue
                    w = value.cpu().data.numpy()
                    f.write(w)
                    print(0, idx, key, value.shape)
                print()

                for idx in range(42, 66):  # BACKBONE(resnet18)
                    key, value = weight_list[idx]
                    if "num_batches_tracked" in key:
                        print(idx, "--------------------")
                        continue
                    w = value.cpu().data.numpy()
                    f.write(w)
                    print(0, idx, key, value.shape)
                print()

                for idx in range(54, 55):  # prelu layer2
                    key, value = weight_list[idx]
                    if "num_batches_tracked" in key:
                        print(idx, "--------------------")
                        continue
                    w = value.cpu().data.numpy()
                    f.write(w)
                    print(0, idx, key, value.shape)
                print()

                for idx in range(66, 84):  # BACKBONE(resnet18)
                    key, value = weight_list[idx]
                    if "num_batches_tracked" in key:
                        print(idx, "--------------------")
                        continue
                    w = value.cpu().data.numpy()
                    f.write(w)
                    print(0, idx, key, value.shape)
                print()

                for idx in range(78, 79):  # prelu layer2
                    key, value = weight_list[idx]
                    if "num_batches_tracked" in key:
                        print(idx, "--------------------")
                        continue
                    w = value.cpu().data.numpy()
                    f.write(w)
                    print(0, idx, key, value.shape)
                print()

                for idx in range(84, 108):  # BACKBONE(resnet18)
                    key, value = weight_list[idx]
                    if "num_batches_tracked" in key:
                        print(idx, "--------------------")
                        continue
                    w = value.cpu().data.numpy()
                    f.write(w)
                    print(0, idx, key, value.shape)
                print()

                for idx in range(96, 97):  #  prelu layer3
                    key, value = weight_list[idx]
                    if "num_batches_tracked" in key:
                        print(idx, "--------------------")
                        continue
                    w = value.cpu().data.numpy()
                    f.write(w)
                    print(0, idx, key, value.shape)
                print()

                for idx in range(108, 126):  # BACKBONE(resnet18)
                    key, value = weight_list[idx]
                    if "num_batches_tracked" in key:
                        print(idx, "--------------------")
                        continue
                    w = value.cpu().data.numpy()
                    f.write(w)
                    print(0, idx, key, value.shape)
                print()

                for idx in range(120, 121):  #  prelu layer3
                    key, value = weight_list[idx]
                    if "num_batches_tracked" in key:
                        print(idx, "--------------------")
                        continue
                    w = value.cpu().data.numpy()
                    f.write(w)
                    print(0, idx, key, value.shape)
                print()

                for idx in range(126, 150):  # BACKBONE(resnet18)
                    key, value = weight_list[idx]
                    if "num_batches_tracked" in key:
                        print(idx, "--------------------")
                        continue
                    w = value.cpu().data.numpy()
                    f.write(w)
                    print(0, idx, key, value.shape)
                print()

                for idx in range(138, 139):  #  prelu layer4
                    key, value = weight_list[idx]
                    if "num_batches_tracked" in key:
                        print(idx, "--------------------")
                        continue
                    w = value.cpu().data.numpy()
                    f.write(w)
                    print(0, idx, key, value.shape)
                print()

                for idx in range(150, 168):  # BACKBONE(resnet18)
                    key, value = weight_list[idx]
                    if "num_batches_tracked" in key:
                        print(idx, "--------------------")
                        continue
                    w = value.cpu().data.numpy()
                    f.write(w)
                    print(0, idx, key, value.shape)
                print()

                for idx in range(162, 163):  #  prelu layer4
                    key, value = weight_list[idx]
                    if "num_batches_tracked" in key:
                        print(idx, "--------------------")
                        continue
                    w = value.cpu().data.numpy()
                    f.write(w)
                    print(0, idx, key, value.shape)
                print()

                for idx in range(168, 173):  # BACKBONE(resnet18)
                    key, value = weight_list[idx]
                    if "num_batches_tracked" in key:
                        print(idx, "--------------------")
                        continue
                    w = value.cpu().data.numpy()
                    f.write(w)
                    print(0, idx, key, value.shape)
                print()

                for idx in range(174, 175):  # BACKBONE(resnet18)
                    key, value = weight_list[idx]
                    w = value.transpose(1,0)
                    w = w.cpu().data.numpy()
                    w = np.ascontiguousarray(w, dtype=np.float32)   # Valueerror ndarray is not C-Contiguous # w.flags['C_CONTIGUOUS'] 로 확인 가능
                    f.write(w)
                    print(0, idx, key, w.shape)
                print()

                for idx in range(175, 180):  # BACKBONE(resnet18)
                    key, value = weight_list[idx]
                    if "num_batches_tracked" in key:
                        print(idx, "--------------------")
                        continue
                    w = value.cpu().data.numpy()
                    f.write(w)
                    print(0, idx, key, value.shape)
                print()

