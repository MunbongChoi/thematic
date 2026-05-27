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
            - .TIF
        - label/
            - .JSON
    - valid/
        - image/
            - .TIF
        - label/
            - .JSON
- infer_data/
- AGENT.md
- train.py
- test.py
- model.py
- infer.py

## 4. 요청사항
train과 test를 진행할 때, model변경이 가능하도록
