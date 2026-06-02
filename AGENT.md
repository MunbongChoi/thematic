# Project Coding Guide

## 1. 프로젝트 목적
이 프로젝트는 Satellite 이미지에서 Panoptic Segmentation을 진행한다.
## 2. 기술 스택

- Language: Python
- Use Model : YOLO26, Unet, SAM 등
- Metrics : IOU, F1, Precision, Recall, mAP50
- input : tif, tiff
- output : GeoJSON

## 3. 프로젝트 구조

- dataset/
    - train/
        - image/
            - *.TIF
        - label/
            - *.JSON
    - test/
        - image/
            - *.TIF
        - label/
            - *.JSON

- model/
    - YOLO26/
        - train.py
        - infer.py
        - model.py
    - UNet/
        - train.py
        - infer.py
        - model.py
    - GeoSAM/
        - train.py
        - infer.py
        - model.py
    - Mask2Former
        - train.py
        - infer.py
        - model.py
- infer_data/
- AGENT.md

## 4. 요청사항
각 모델을 학습하고 테스트 할 수 있도록 설계
모델 각각 파일 생성
멀티GPU 사용
데이터 처리부분 따로 구분할 것

## 5. 필수사항
GeoJSON에는 면적, 좌표, 좌표계, 감지한 객체의 id당 class가 들어가야한다.
좌표계는 사용자가 입력한다.
면적은 GSD별로 자동으로 계산되어야한다.
