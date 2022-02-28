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

#### 2022.02.27 model 120
CenterCrop((299, 299)),<br>
Resize(resize, Image.BILINEAR),<br>
AutoAugment(AutoAugmentPolicy.SVHN),<br><br>
Best Score: 11Epoch [Val] acc : 74.58%, loss: 0.72 || best acc : 84.37%, best loss: 0.49<br>
<br>
제출 결과: f1:0.5446    acc: 66.0952<br>

#### model 121
RandomCrop((100, 100)) 추가<br>
Epoch 100회 실시<br>
결과: 망함<br>
<br>
<br>

#### model 123<br>
RandomHorizontalFlip(p=0.5) 추가<br>
Epoch 200회 실시<br>
Best Score: 141Epoch [Val] acc : 88.12%, loss: 0.51 || best acc : 88.17%, best loss: 0.47 <br>
제출 결과: f1:0.6119	acc:70.8413<br>
<br>

#### model 124<br>
modedl#123에서 f1 Loss로 변경<br>
10 epoch로는 acc 향상이 크지 않음<br>
<br>

#### model 125<br>
50epoch 진행<br>
<br>

#### model 126<br>
250epoch 진행<br>
Best Score: 219Epoch [Val] acc : 84.81%, loss: 0.35 || best acc : 85.82%, best loss: 0.34
제출 결과: 0.5622	67.1905<br>
해당 모델에서는 Cross Enropy Loss가 나은 것으로 판단 됨<br>
<br>