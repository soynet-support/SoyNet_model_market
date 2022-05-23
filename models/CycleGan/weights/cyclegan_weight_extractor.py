import torch
import argparse
import numpy as np

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.determinstic = False

def weight_extractor(output_path, load_path) :
    # 웨이트 불러오기
    weights = torch.load(load_path)

    weight_list = [(key, value) for (key, value) in weights.items()]

    with open(output_path, 'wb') as f:
        dumy = np.array([0] * 10, dtype=np.float32)
        dumy.tofile(f)

        for idx in range(0, len(weight_list)):
            key, w = weight_list[idx]
            if any(skip in key for skip in ('num_batches_tracked', 'running')):
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
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="웨이트 추출 옵션 설정")
    parser.add_argument("-w", "--weight_path", help="저장할 소이넷 웨이트 파일 경로", default="cyclegan.weights" ) # cyclegan, fanogan, fcn, i_resnet_v2
    parser.add_argument("-l", "--load_path", help="변환할 파이토치, 텐서플로 웨이트 경로", default='./checkpoints/style_vangogh_pretrained/latest_net_G.pth')

    args = parser.parse_args()

    weight_path = args.weight_path  # 생성할 소이넷 웨이트 파일 경로
    load_path = args.load_path

    # weights = torch.load(load_path, map_location=str('cuda:0'))


    weight_extractor(weight_path, load_path) # weight 추출기






