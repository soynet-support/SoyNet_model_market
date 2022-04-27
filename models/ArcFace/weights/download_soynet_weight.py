import wget

if __name__ == '__main__':
    weight_url = [
        'https: // kr.object.iwinv.kr / model_market_weight / arc_face_r100.weights',
        'https: // kr.object.iwinv.kr / model_market_weight / arc_face_r18.weights',
        'https: // kr.object.iwinv.kr / model_market_weight / arc_face_r50.weights'
    ]

    for url in weight_url:
        wget.download(url, './')
