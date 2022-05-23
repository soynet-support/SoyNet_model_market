model.load_state_dict(torch.load(CFG.model_path))
model = model.to(CFG.device)

if 0:
    weights = model.state_dict()
    weight_list = [(key, value) for (key, value) in weights.items()]
    newfile = open('weight_structure_eca_nfnet_l0.txt', 'w', encoding='utf-8')
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

if 0:
    weight_path = "eca_nfnet_l0.weights"
    weights = model.state_dict()
    weight_list = [(key, value) for (key, value) in weights.items()]

    with open(weight_path, 'wb') as f:
        dumy = np.array([0] * 10, dtype=np.float32)
        dumy.tofile(f)

        for idx in range(0, len(weight_list)):
            key, w = weight_list[idx]

            if any(skip in key for skip in ('num_batches_tracked', 'gain')):
                print(idx, "--------------------")
                continue

            if "conv" in key and "weight" in key:
                key_gain, gain_w = weight_list[idx + 2]
                if "gain" in key_gain:  # ScaledStdConv2d: gain이 있는지 없는지 확인
                    eps = 1e-5
                    gamma = 1.7881293296813965
                    scale = gamma * w[0].numel() ** -0.5
                    out_channels = w.shape[0]

                    w = F.batch_norm(
                        w.view(1, out_channels, -1), None, None,
                        weight=(gain_w * scale).view(-1),
                        training=True, momentum=0., eps=eps).reshape_as(w)
                    w = w.cpu().data.numpy()
                else:
                    w = w.cpu().data.numpy()

            else:
                if len(w.shape) == 2:
                    print("transpose() \n")
                    w = w.transpose(1, 0)
                    w = w.cpu().data.numpy()
                else:
                    w = w.cpu().data.numpy()
            w.tofile(f)
            print(0, idx, key, w.shape)