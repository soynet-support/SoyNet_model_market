import wget

if __name__ == '__main__':
    weight_url = [
        'https://kr.object.iwinv.kr/model_market_weight/retina_face_r50.weights'
    ]

    for url in weight_url:
        wget.download(url, './')
