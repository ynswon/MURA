# MURA
MURA(musculoskeletal radiographs) - bone x-ray

Reference: https://stanfordmlgroup.github.io/competitions/mura/

### Prerequite
Python 3.5
<br>TensorFlow 1.8+
<br>keras 2.2.0
<br>numpy 1.14.5
<br>pandas 0.23.3
<br>sklearn 0.19.1

### histogram_equalization usage
keras_preprocessing package 의 image.py 파일에 data augmentation부분이 수정되야함
/home/casper/.local/lib/python3.5/site-packages/keras_preprocessing/image.py 를 해당 image.py로 교체 

```shell
### Clone the repo.
git clone https://github.com/AItrics/MURA.git
<br>cd MURA

### Transform MURA-v1.1 folder to data folder
python3 download_and_convert_mura.py
<br>-> training에 들어가는 Input형태로 폴더와 파일을 정라하여 /data폴더에 넣어줌

### Run
python3 train.py

### To evaluate
python3 eval.py
python3 predict.py MURA-v1.1/valid.csv prediction.csv
```
### Performance 
- Ensemble Model
(model1 + model3 + model3  + model5 + model5) /5 로 평균낸 ensemble

|                                  | val_loss  | accuracy  |
| -------------------------------: | :-------- | :---------|
|  DenseNet201(420x420)            | 0.4320    | 0.8332    |
|  DenseNet169(520x520)with cutout | 0.4045    | 0.8313    |
|  InceptionResNetV2(420x420)      | 0.4211    | 0.8341    |
|  DenseNet201(420x420)            | 0.4311    | 0.8177    |
|  DenseNet169(520x520)            | 0.4082    | 0.8307    |

|                 | per image | per study |
| --------------: | :-------- | :---------|
|    accuracy     | 0.831     | 0.857     |
|       f1        | 0.846     | 0.875     |
|    precision    | 0.806     | 0.840     |
|     recall      | 0.889     | 0.914     |
| **cohen_kappa** | **0.661** | **0.708** |


- Single Model (DenseNet169 with Histogram Equalization, batch_size=8,  img_size=420)
@strange://shared/casper/MURA/models/DenseNet169_420_NEW_HIST.hdf5

|                 | single    | 
| --------------: | :-------- | 
|      loss       | 0.4125    | 
|    accuracy     | 0.856     | 
|       f1        | 0.870     | 
|    precision    | 0.866     | 
|     recall      | 0.873     | 
| **cohen_kappa** | **0.705** | 

### data
데이터경로 :
<br>MURA-v1.1 : @strange:/shared/casper/MURA/MURA-v1.1(36808장) 또는 @strange:/shared/casper/MURA/MURA-v1.1.zip 
<br>MURA-v1.0 : @strange:/shared/casper/MURA/MURA-v1.0          또는 @strange:/shared/casper/MURA/mura-v1.0.zip
<br>MURAv1.0 + MURAv1.1(49158장) : @strange:/shared/casper/MURA/data 

### log
로그 보기 예시:
tensorboard --logdir=./logs/DenseNet169_420_NEW_HIST/

### model
모델 경로:
./models
