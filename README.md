# MURA
MURA(musculoskeletal radiographs) - bone x-ray

Reference: https://stanfordmlgroup.github.io/competitions/mura/

Prerequisite
Python 3.5
TensorFlow 1.8+
keras 2.2.0
numpy 1.14.5
pandas 0.23.3
sklearn 0.19.1

How To Run
#Modification(histogram_equalization)
keras_preprocessing package 의 image.py 파일에 data augmentation부분이 수정되야함
/home/casper/.local/lib/python3.5/site-packages/keras_preprocessing/image.py

# Clone the repo.
git clone https://github.com/AItrics/MURA.git
cd MURA-v1.1

# Transform MURA-v1.1 oflder to data folder
python3 download_and_convert_mura.py
->training에 들어가는 Input형태로 폴더와 파일을 정라하여 /data폴더에 넣어줌

# Run! 
CUDA_VISIBLE_DEVICES=0 python3 train.py

# To evaluate
CUDA_VISIBLE_DEVICES=0 python3 eval.py
CUDA_VISIBLE_DEVICES=0 python3 predict.py MURA-v1.1/valid.csv prediction.csv

Ensemble Model
(model1 + model3 + model3  + model5 + model5) /5 로 평균낸 ensemble

                     validation loss	accuracy	
DenseNet201(420x420)	0.4320	0.8332	model1
DenseNet169(520x520)with Random Erase	0.4045	0.8313	model2
InceptionResNetv2(420x420)	0.4211	0.8341	model3
DenseNet210 (420x420)	0.4311	0.8177	model4
DenseNet169 (520x520)	0.4082	0.8307	model5


per image metric	per study metric
accuracy	0.831	0.857
f1	0.846	0.875
precision	0.806	0.840
recall	0.889	0.914
cohen_kappa	0.661	0.708

Single Model (DenseNet169 with Histogram Equalization, batch_size=8,  img_size=420)
@stark://shared/casper/MURA/models/DenseNet169_420_NEW_HIST.hdf5
loss	0.4125
accuracy	0.856
f1	0.870
precision	0.866
recall	0.873
cohen_kappa	0.705

MURA-V1.1 validation data 기준 0.705

데이터경로 :
MURA-v1.1 : /shared/casper/MURA/MURA-v1.1(36808장) 또는 /shared/casper/MURA/MURA-v1.1.zip 
MURA-v1.0 : /shared/casper/MURA/MURA-v1.0          또는 /shared/casper/MURA/mura-v1.0.zip
MURAv1.0 + MURAv1.1(49158장) : /shared/casper/MURA/data 

로그 보기 예시:
tensorboard --logdir=./logs/DenseNet169_420_NEW_HIST/
모델 경로:
./models
