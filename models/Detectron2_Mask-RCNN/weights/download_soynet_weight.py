import wget

if __name__ == '__main__':
    weight_url = [
        'https://kr.object.iwinv.kr/model_market_weight/maskrcnn-x101-fpn-detectron2.weights',
        'https://kr.object.iwinv.kr/model_market_weight/maskrcnn-r101-fpn-detectron2.weights'
    ]

    for url in weight_url:
        wget.download(url, './')
