
## 대회내용      
세종테크노파크에서는 세종 자율주행 시범운행지구 운영을 통해 자율주행 차량 데이터와
관제 데이터를 보유하고 있습니다.        

 2024년 경진대회에서는 도로 돌발 상황 관제 데이터를
활용 횡단, 무단횡단 보행자를 감지하는 AI 경진대회를 진행합니다.


### 이미지 입력 방식

```c
python excute.py --image_dir '입력 이미지 폴더'
```
이미지 입력 시, 'data'폴더에 횡단보도 보행자, 무단횡단 보행자 라벨이 입력된다. 



#### requirements.txt
```c
pip install segmentation_models_pytorch
pip install ultralytics     
pip install albumentations
pip install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install scikit-learn
pip install matplotlib
```

#### 총 **4개의 폴더**로 이루어져 있다.

1. 입력 이미지 데이터 폴더 
    -   '상대경로'/evaluationData
2. 인도 / 도로 식별 Segmentation 폴더
    -   '상대경로'/Last_Mask
3. 모델파일 보관 폴더
    -   '상대경로'/model


3. 정답 예측 폴더
    -   '상대경로'/data




