import wget

if __name__ == '__main__':
    weight_url = [
        'https://kr.object.iwinv.kr/model_market_weight/efficientnet-b0_noisy-student.weights',
        'https://kr.object.iwinv.kr/model_market_weight/efficientnet-b1_noisy-student.weights',
        'https://kr.object.iwinv.kr/model_market_weight/efficientnet-b2_noisy-student.weights',
        'https://kr.object.iwinv.kr/model_market_weight/efficientnet-b3_noisy-student.weights',
        'https://kr.object.iwinv.kr/model_market_weight/efficientnet-b4_noisy-student.weights',
        'https://kr.object.iwinv.kr/model_market_weight/efficientnet-b5_noisy-student.weights',
        'https://kr.object.iwinv.kr/model_market_weight/efficientnet-b6_noisy-student.weights',
        'https://kr.object.iwinv.kr/model_market_weight/efficientnet-b7_noisy-student.weights'
    ]

    for url in weight_url:
        wget.download(url, './')
