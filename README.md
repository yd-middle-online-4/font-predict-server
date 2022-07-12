# 서체 폰트 판별기 (베이스라인 모델)

클라이언트 서버로 온 이미지 데이터를 분석하여 결과를 내보내줍니다.

## Requirement(필수)

- numpy
- flask
- flask_cors
- tensorflow (또는 tensorflow-cpu)

## 사용법

- 모델 서버를 작동시킵니다
  - python server.py
- [프론트엔드](https://62cbfd8e01f49e080a309a8d--font-predict.netlify.app/) 페이지
- 페이지의 '서버 주소'란에 가동중인 주소를 입력하고 적용을 눌러줍니다
  - ex) http://127.0.0.1:5555/predict
  - http://127.0.0.1:"포트번호" <- 중요
- 서버의 /predict 라우터에 request를 요청하므로 잘 확인해주세요
- 이미지를 업로드하고 예시와 비슷하게 맞춰주세요
- Submit을 누르고 분석 결과를 봅니다
