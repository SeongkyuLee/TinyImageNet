# Tiny Imagenet
UCSB에서 수강한 고급 컴퓨터 비전 수업의 최종 프로젝트입니다.
100개 라벨을 가진 50,000개의 이미지(training, validation)를 학습하여 라벨이 없는 10,000개의 이미지(testing)의 라벨을 추측하는 프로젝트입니다. 
프로젝트에 사용된 이미지는 이미지넷 원본 이미지(224x224)보다 작은 크기의 이미지(56x56)를 사용 하였습니다.

# 환경
운영 체제 : Ubuntu 16.04 LTS

가상 환경 : Microsoft Azure 

라이브러리 : Pytorch

알고리즘 : CNN

아키텍처 : VGG Net 19 Layer, Res Net 34 Layer

# 과정
1) 라벨이 붙은 50,000개의 이미지를 40,000개의 training 데이터와 10,000개의 validation 데이터로 나눕니다.

2) Overfitting를 피하기 위해 40,000개의 training 데이터를 data augmentation 합니다. 

3) 이전 epoch와 비교해서 accuracy가 큰 차이가 없을 때까지 반복 학습 시킵니다. 4번째 학습마다 overfitting를 피하기 위해 learning rate를 감소시킵니다.

4) Ensemble 방식으로 validation 데이터를 평가합니다. 예를 들어, ResNet 34 아키텍처로 학습한 모델1과 모델2를 만듭니다. 모델1이 한 이미지에 대해서 개(0.5), 고양이(0.5), 모델2가 개(0.7), 고양이(0.3)이란 결과를 보여줬다면, 절대값이 가장 큰 모델2의 개(0.7)를 그 사진의 라벨로 정합니다.

5) 가장 높은 accuracy 값을 가진 아키텍처(VGG Net 또는 Res Net)에 10,000개의 validation 데이터도 추가 학습을 시키고, 라벨이 없는 10,000개의 testing 이미지에 라벨을 붙입니다.

# 모델 선택 과정
1) VGG Net vs Res Net
ResNet이 VGGNet보다 약 7% 더 높은 accuracy를 보여줍니다. 둘 다 pre-trained 되지 않은 상태에서 VGGNet은 38% 정도의 accuracy에 수렴하고, Res Net은 45% 정도의 accuracy에 수렴합니다. 

VGG Net 19 (non-pretrained, non-ensemble)
<img src="https://github.com/SeongkyuLee/TinyImageNet/blob/master/figure/VGG19_no_pre_no_ens.jpg"/>

Res Net 34 (non-pretrained, non-ensemble)
<img src="https://github.com/SeongkyuLee/TinyImageNet/blob/master/figure/Res34_no_pre_no_ens.jpg?"/>

2) pre-trained Res Net vs non-pre-trained Res Net
Pre-trained된 ResNet이 약 35% 더 높은 accuracy를 보여줍니다. pre-trained된 모델은 80%의 accuracy를 보여주고, non-pre-trained된 모델은 45% 정도의 accuracy를 보여줍니다. 

Non-pretrained Res Net 34 (non-pretrained, non-ensemble)
<img src="https://github.com/SeongkyuLee/TinyImageNet/blob/master/figure/Res34_pre_no_ens.jpg"/>

Pretrained Res Net 34 (pretrained, non-ensemble)
<img src="https://github.com/SeongkyuLee/TinyImageNet/blob/master/figure/Res34_no_pre_no_ens.jpg"/>

3) ensemble model vs non-ensemble model
ensemble을 사용한 모델이 약 2% 정도 더 높은 accuracy를 보여줍니다. ensemble을 사용한 모델은 47% 정도의 accuracy를 보여주고, ensemble을 사용하지 않은 모델은 45% 정도의 accuracy를 보여줍니다. 

Non-ensemble Res Net 34 (pretrained, non-ensemble)
<img src="https://github.com/SeongkyuLee/TinyImageNet/blob/master/figure/Res34_pre_no_ens.jpg"/>

Ensemble Res Net 34 (pretrained, ensemble)
<img src="https://github.com/SeongkyuLee/TinyImageNet/blob/master/figure/Res34_pre_ens.jpg"/>

# 모델 선택 결과
non-ensemble pre-trained ResNet 34 Layer를 선택 했습니다. ensemble 모델을 사용하기 위해서는 2개의 모델을 만들어야 하므로, 기존 학습 시간의 2배가 된다는 점을 생각하면, 제한된 프로젝트 시간에서는 효율적이지 않은 방식이라 판단이 되어서 non-ensemble model를 선택 했습니다.

