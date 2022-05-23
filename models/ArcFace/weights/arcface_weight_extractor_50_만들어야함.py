
from __future__ import print_function
import os
import cv2
from models1.resnet_custom import resnet_face50
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
    image = cv2.imread(img_path, 1) # BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR -> RGB
    if image is None:
        return None
    image = np.stack((image, np.fliplr(image)), axis = 0)
    image = image.transpose((0, 3, 1, 2))
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
    model = resnet_face50((3, 128, 128), use_se=False)

    # 모델 불러오기
    model = DataParallel(model)
    model.load_state_dict(torch.load('./checkpoints/resnet_face100_35.pth'))
    model.to(torch.device("cuda"))
    model.eval()

    weight_path = "arc_face_r100_MS1M_35.weights"

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

            f.write(np.array([0] * 10, dtype=np.float32))  # dummy 10 line

            if 0:  # 전체 보기
                for idx in range(len(weight_list)):
                    key, value = weight_list[idx]
                    if "num_batches_tracked" in key:
                        print(idx, "--------------------")
                        continue
                    print(idx, key, value.shape)
                exit()

            if 1:
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
                    w.tofile(f)
                    print(0, idx, key, w.shape)

                print("Weight Extract Done!")
                exit(-1)

