# net and model
net = load_model(net, './weights/Resnet50_Final.pth', False)
net.eval()
device = torch.device("cuda")
net = net.to(device)

weight_path = "E:/DEV4/mgmt/weights/retina_face_r50.weights"

if 0:  # weight download, (0 -> off, 1 -> on)
    print()
    with open(weight_path, 'wb') as f:
        f.write(np.array([0] * 10, dtype=np.float32))  # dummy 10 line

        weights = net.state_dict()
        weight_list = [(key, value) for (key, value) in weights.items()]

        if 0:  # 전체 보기
            for idx in range(len(weight_list)):
                key, value = weight_list[idx]
                if "num_batches_tracked" in key:
                    print(idx, "--------------------")
                    continue
                print(idx, key, value.shape)
            exit()

        for idx in range(0, 335):  # BACKBONE(resnet50 + fpn)
            key, value = weight_list[idx]
            if "num_batches_tracked" in key:
                print(idx, "--------------------")
                continue
            w = value.cpu().data.numpy()
            f.write(w)
            print(0, idx, key, value.shape)
        print()

        for idx in range(341, 347):  # merge2
            key, value = weight_list[idx]
            if "num_batches_tracked" in key:
                print(idx, "--------------------")
                continue
            w = value.cpu().data.numpy()
            f.write(w)
            print(0, idx, key, value.shape)
        print()

        for idx in range(335, 341):  # merge1
            key, value = weight_list[idx]
            if "num_batches_tracked" in key:
                print(idx, "--------------------")
                continue
            w = value.cpu().data.numpy()
            f.write(w)
            print(0, idx, key, value.shape)
        print()

        for idx in range(347, 438):  # ssh
            key, value = weight_list[idx]
            if "num_batches_tracked" in key:
                print(idx, "--------------------")
                continue
            w = value.cpu().data.numpy()
            f.write(w)
            print(0, idx, key, value.shape)
        print()

        for idx in range(444, 450):  # bboxHead
            key, value = weight_list[idx]
            if "num_batches_tracked" in key:
                print(idx, "--------------------")
                continue
            w = value.cpu().data.numpy()
            f.write(w)
            print(0, idx, key, value.shape)
        print()

        for idx in range(438, 444):  # ClassHead
            key, value = weight_list[idx]
            if "num_batches_tracked" in key:
                print(idx, "--------------------")
                continue
            w = value.cpu().data.numpy()
            f.write(w)
            print(0, idx, key, value.shape)
        print()

        for idx in range(450, 455):  # LandmarkHead
            key, value = weight_list[idx]
            if "num_batches_tracked" in key:
                print(idx, "--------------------")
                continue
            w = value.cpu().data.numpy()
            f.write(w)
            print(0, idx, key, value.shape)
        print()