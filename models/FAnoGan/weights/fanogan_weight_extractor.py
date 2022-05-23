import torch
import argparse
import numpy as np
from networks import Generator, Encoder

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.determinstic = False

def weight_extractor(output_path, encoder_path, generator_path) :
    # 웨이트 불러오기
    weights = torch.load(encoder_path)
    weights1 = torch.load(generator_path)

    weight_list_E = [(key, value) for (key, value) in weights.state_dict().items()] # fanogan,
    weight_list_G = [(key, value) for (key, value) in weights1.state_dict().items()]
    # if (model_type == "fanogan"):
    #     weight_list = [(key, value) for (key, value) in weights['model_state_dict'].items()] # fanogan
    if 0:
        # Encoder
        for idx in range(len(weight_list_E)):
            key, w = weight_list_E[idx]
            w = w.cpu().numpy()
            if "num_batches_tracked" in key:
                print(idx, "--------------------")
                continue
            print(0, idx, key, w.shape)

        # Generator
        print("============== Generator ==============")
        for idx in range(len(weight_list_G)):
            key, w = weight_list_G[idx]
            w = w.cpu().numpy()
            if "num_batches_tracked" in key:
                print(idx, "--------------------")
                continue
            print(0, idx, key, w.shape)
        exit(-1)

    if 1:
        with open(output_path, 'wb') as f:
            dumy = np.array([0] * 10, dtype=np.float32)
            dumy.tofile(f)

            #Encoder
            for idx in range(0, len(weight_list_E)):
                key, w = weight_list_E[idx]
                if any(skip in key for skip in ('num_batches_tracked', 'bn.weight')): # inception_resnet_v2 : bn.weight'
                    continue

                if len(w.shape) == 2:
                    print("transpose() \n")
                    w = w.transpose(1, 0)
                    w = w.cpu().data.numpy()
                else:
                    w = w.cpu().data.numpy()
                w.tofile(f)
                print(0, idx, key, w.shape)

            # Generator
            print(idx, "----------Generator----------")
            for idx in range(0, len(weight_list_G)):
                key, w = weight_list_G[idx]
                if any(skip in key for skip in ('num_batches_tracked', 'bn.weight')):  # inception_resnet_v2
                    print(idx, "--------------------")
                    continue

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

    print('웨이트 생성 완료')
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="웨이트 추출 옵션 설정")
    parser.add_argument("-w", "--weight_path", help="저장할 소이넷 웨이트 파일 경로",type=str,  default="fanogan.weights" ) # cyclegan, fanogan, fcn, i_resnet_v2
    parser.add_argument("-e", "--encoder_path", help="변환할 파이토치, 텐서플로 웨이트 경로", type=str, default='./model/DCGAN_AE_E.pth')
    parser.add_argument("-g", "--generator_path", help="변환할 파이토치, 텐서플로 웨이트 경로", type=str, default='./model/DCGAN_AE_G.pth')
    args = parser.parse_args()

    weight_path = args.weight_path  # 생성할 소이넷 웨이트 파일 경로
    encoder_path = args.encoder_path
    generator_path = args.generator_path
    # weights = torch.load(load_path, map_location=str('cuda:0'))


    weight_extractor(weight_path, encoder_path, generator_path) # weight 추출기





