# Project Coding Guide

## 1. 프로젝트 목적
이 프로젝트는 Satellite 이미지에서 ROAD Extraction 해야한다.

## 2. 기술 스택

- Language: Python
- Use Model : YOLO26, Unet, SAM 등

## 3. 프로젝트 구조

- dataset/
    - train/
        - image/
            - *.TIF
        - label/
            - *.JSON
    - valid/
        - image/
            - *.TIF
        - label/
            - *.JSON
- infer_data/
- AGENT.md
- train.py
- test.py
- infer.py

## 4. 요청사항
train과 test를 진행할 때, model변경이 가능하도록
모델 각각 파일 생성
멀티GPU 사용
데이터 처리부분 따로 구분할 것
모델 전부 학습 가능하게 만들 것
모델 추론부분 만들 것
yolo26, mask2former, Unet, SAM 사용가능하게 할 것
