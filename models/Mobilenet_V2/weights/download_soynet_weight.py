import wget

if __name__ == '__main__':
    weight_url = [
        'https://kr.object.iwinv.kr/model_market_weight/mobilenet_v2.weights'
    ]

    for url in weight_url:
        wget.download(url, './')
