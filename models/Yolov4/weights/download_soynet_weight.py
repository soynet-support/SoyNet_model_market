import wget

if __name__ == '__main__':
    weight_url = [
        'https://kr.object.iwinv.kr/model_market_weight/yolov4.weights',
        'https://kr.object.iwinv.kr/model_market_weight/yolov4-tiny.weights'
    ]

    for url in weight_url:
        wget.download(url, './')
