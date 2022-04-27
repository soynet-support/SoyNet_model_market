import wget

if __name__ == '__main__':
    weight_url = [
        'https://kr.object.iwinv.kr/model_market_weight/ssd_mobilenet_v2_fpnlite_640x640.weights'
    ]

    for url in weight_url:
        wget.download(url, './')
