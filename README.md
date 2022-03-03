# pstage_01_image_classification

## Getting Started    
### Dependencies
- torch==1.7.1
- torchvision==0.8.2                                                              

### Install Requirements
- `pip install -r requirements.txt`

### Training
- `SM_CHANNEL_TRAIN={YOUR_TRAIN_IMG_DIR} SM_MODEL_DIR={YOUR_MODEL_SAVING_DIR} python train.py`

### Inference
- `SM_CHANNEL_EVAL={YOUR_EVAL_DIR} SM_CHANNEL_MODEL={YOUR_TRAINED_MODEL_DIR} SM_OUTPUT_DATA_DIR={YOUR_INFERENCE_OUTPUT_DIR} python inference.py`

### Evaluation
- `SM_GROUND_TRUTH_DIR={YOUR_GT_DIR} SM_OUTPUT_DATA_DIR={YOUR_INFERENCE_OUTPUT_DIR} python evaluation.py`


#### 2022.02.26 model118
Inception V3 Model로 변경<br>
f1: 0.5119 accuracy:65.2857<br>
동일조건에서 resnet을 사용했을 때 보다 대폭 상승<br>

### 2022-02-27
### model 120
CenterCrop((299, 299)),<br>
Resize(resize, Image.BILINEAR),<br>
AutoAugment(AutoAugmentPolicy.SVHN),<br><br>
Best Score: 11Epoch [Val] acc : 74.58%, loss: 0.72 || best acc : 84.37%, best loss: 0.49<br>
<br>
제출 결과: f1:0.5446    acc: 66.0952<br>

### model 121
RandomCrop((100, 100)) 추가<br>
Epoch 100회 실시<br>
결과: 망함<br>
<br>
<br>

### model 123<br>
RandomHorizontalFlip(p=0.5) 추가<br>
Epoch 200회 실시<br>
Best Score: 141Epoch [Val] acc : 88.12%, loss: 0.51 || best acc : 88.17%, best loss: 0.47 <br>
제출 결과: f1:0.6119	acc:70.8413<br>
<br>

### model 124<br>
modedl#123에서 f1 Loss로 변경<br>
10 epoch로는 acc 향상이 크지 않음<br>
<br>

### 2022-02-28
### model 125<br>
50epoch 진행<br>
<br>

### model 126<br>
250epoch 진행<br>
Best Score: 219Epoch [Val] acc : 84.81%, loss: 0.35 || best acc : 85.82%, best loss: 0.34<br>
제출 결과: 0.5622	67.1905<br>
해당 모델에서는 Cross Enropy Loss가 나은 것으로 판단 됨<br>
<br>
<br>

### model 132<br>
Inception-Resnet-V2 Model로 변경<br>
82Epoch 중단<br>
Best Score: 63Epoch [Val] acc : 86.22%, loss: 0.51 || best acc : 86.85%, best loss: 0.48<br>
제출 결과: 0.5741	68.4603<br>
더 진행해볼 필요?<br>

### model 133<br>
200Epoch 진행<br>
Best Score: 132Epoch [Val] acc : 87.46%, loss: 0.48 || best acc : 87.46%, best loss: 0.48<br>
큰 향상은 없음<br>
<br>
<br>

### model 134<br>
train data 최대화 필요 → train/val ration 9:1로 변경<br>
50Epoch 진행<br>

### model 135<br>
train data 최대화 필요 → train/val ration 9:1로 변경<br>
150Epoch 진행<br>
제출 결과: 0.5892	69.4286<br>
소요 시간 대비 효율 낮음 → 다시 InceptionV3로 변경<br>
유효한 Augmentation 및 Loss 실험 필요<br>

### model 138<br>
Best Score: 141Epoch [Val] acc : 87.75%, loss: 0.51 || best acc : 88.17%, best loss: 0.47<br>

### model 139<br>
Gray Scale 적용 50Epoch<br>
<br>
Test Dataset: CenterCrop 적용<br>

### model 142<br>
Gaussian noise 추가<br>
Test Dataset: Gray Scale 적용<br>
효율은 그다지 없음