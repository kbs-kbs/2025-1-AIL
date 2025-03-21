# CNN의 목적
모델에 적합한 필터 찾기

이미지가 필터를 통과하면 특징이 나옴

[[-1, 0, 1],
[-1, 0, 1],
[-1, 0, 1]]

예를 들어 위와 같은 필터를 거치면 왼쪽부터 오른쪽으로 밝아지는 특징을 추출
예전에는 필터를 직접 설계

CNN: 필터를 어떻게 설계할 것인가


32x32 이미지에서 16x16 이미지(특징이 추출된) feature map ...-> 4x4 정도에서 평탄화

왜 MLP(Multi Layer Perceptron) 대신에 CNN을 사용할까?
- MLP는 이미지 크기가 커짐에 따라 계산량이 커짐 → 비효율적!
- MLP는 이미지를 1D vector로 변환하여 입력받음 → 공간적 정보(위치, 패턴) 손실
  - CNN은 이미지 속에서 사물의 위치나 크기가 달라져도 분류할 수 있음
 
MLP
- 1000x840 크기의 이미지는 840만 개의 픽셀(입력 뉴런) 필요
- MLP를 사용하면 모든 뉴런이 완전 연결(fully connected) → 엄청난 수의 가중치(파라미터) 필요!
- 연산량이 너무 많아 학습이 어렵고, overfitting 우려가 있음
CNN
- CNN은 전체 이미지를 직접 연결하지 않고, 작은 3×3필터(5x5, …)를 사용하여 이동하면서특징을추출함
- 작은 크기의 필터를 사용해도 모든 영역을 학습할 수 있음
- 학습해야 할 가중치(파라미터) 수가 크게 줄어듦 → 학습 속도 향상, 연산량 감소

결론: CNN은 큰 이미지도 작은 필터로 처리할 수 있으므로 더 적은 파라미터로 학습 가능함

• 합성곱(Convolution)
• 커널과 필터(Kernel & Filter)
• Stride: 커널이 몇 칸씩 움직일 것인가
• 특징맵(Feature Map): 커널을 거친 이미지 맵, convolution 연산으로부터 얻어지는 이미지.
• 패딩(Padding): 이미지 크기가 줄어들지 않게?
• Max Pooling: 가장 큰값으로 축소

RGB의 경우 커널이 3개가 필요함 이 경우에는 3커널이 1필터가 됨

커널을 쓰면 이미지 크기에 따라 특정 비율을 커널로 해상도에 상관없이 특징을 추출해낼 수 있음

자료 14 오류 2*2 -> 3*3 

패딩: 3x3에 패딩을 줘서 5x5로 만들면 3*3커널로 컨볼루션해도 결과가 3x3이 되어 이미지 크기가 유지됨.
왜 마진이 아닌가

패딩의 종류:
- 제로패딩
- 레플리케이션 패딩
- 리플렉션 패딩

max pooling

max pooling을 위한 커널 사용 원본과 다운 샘플링할 크기에 따라
커널 크기와 stride가 정해짐

장점: 연산량 줄어듦
불필요한 정보 제거되어 과적합 방지
객체 탐지에 효과적. 이미지 상의 어느곳에 위치해있는가?

단점: 정보손실

빨간색 자동차 그림이 있어
과적합되어 빨간색 이외의 자동차를 인식을 못하게 될 수 있기 때문에 RGB 정규화
but 색이 중요한 객체라면 안하는게 좋을 듯 또는 정규화하지 않고 데이터셋을 늘리는 방법이 가장 좋을듯



## 사용 언어
|언어|버전|
|---|---|
|Python|3.11.11|

## 사용 라이브러리
|언어|라이브러리|버전|모듈/클래스|용도|
|---|---|---|---|---|
|Python|scikit-learn|1.6.1|sklearn.linear_model/LinearRegression|선형 회귀 모델 사용|
||||sklearn.neighbors/KNeighborsRegressor|k-최근접 이웃 회귀 모델 사용|
||torchvision|0.21.0+cu124|datasets.cifar.CIFAR10|CIFAR10 데이터셋 사용|
||torchvision|0.21.0+cu124|transforms.ToTensor|이미지를 파이토치 텐서로 변환|
||torchvision|0.21.0+cu124|transforms.Compose|이미지 데이터 전처리 함수|
||torchvision|0.21.0+cu124|transforms.RandomHorizontalFlip|데이터셋 증강을 위해 좌우대칭할 때 사용|
||torchvision|0.21.0+cu124|transforms.RandomCrop|객체 위치 조정을 위해 이미지를 랜덤으로 자름|
||pandas|2.2.3|pandas|데이터 불러오기|
||matplotlib|3.10.0|matplotlib.pyplot.plt|데이터 시각화|

## 코드
```
from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# CIFAR10 데이터셋을 불러옴
training_data = CIFAR10(
    root="./", # 현재 디렉토리에 데이터셋을 불러옴
    train=True,
    download=True,
    transform=ToTensor()) # 이미지를 파이토치 텐서로 변환해줌

test_data = CIFAR10(
    root="./",
    train=False,
    download=True,
    transform=ToTensor())

for i in range(9):
    plt.subplot(3, 3, i+1) # 3x3의 몇번째 플롯인지 지정 (1번부터 시작)
    plt.imshow(training_data.data[i]) # 트레이닝 데이터의 이미지를 보여줌
    plt.title(training_data.classes[training_data.targets[i]]) # 트레이닝 데이터의 label 표시
    plt.axis('off')  # 축 안 보이게
plt.tight_layout()
plt.show() # 플롯 집합 보이기
```

```
#데이터 증강
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomCrop

transforms = Compose([ # 데이터 전처리 함수를 나열하여 한번에 적용
   T.ToPILImage(), # numpy 행렬을 PIL 이미지로 변환
   RandomCrop((32, 32), padding=4), # 4픽셀 패딩 후 32x32로 랜덤 자르기
   RandomHorizontalFlip(p=0.5), # y축으로 기준으로 대칭, 50% 확률로
])

training_data = CIFAR10(
    root="./",
    train=True,
    download=True,
    transform=transforms) # transform에는 데이터를 변환하는 함수가 들어감

test_data = CIFAR10(
    root="./",
    train=False,
    download=True,
    transform=transforms)

for i in range(9):
   plt.subplot(3, 3, i+1)
   plt.imshow(transforms(training_data.data[i]))
plt.show()
```

```
import torch

#이미지 정규화를 하기 위한 사전 작업

training_data = CIFAR10(
    root="./",
    train=True,
    download=True,
    transform=ToTensor())

# item[0]은 이미지, item[1]은 정답 레이블 -> 이미지만 추출
imgs = [item[0] for item in training_data]

# imgs를 하나로 합침
imgs = torch.stack(imgs, dim=0).numpy()

# rgb 각각의 평균
mean_r = imgs[:,0,:,:].mean()
mean_g = imgs[:,1,:,:].mean()
mean_b = imgs[:,2,:,:].mean()
print(mean_r,mean_g,mean_b)

# rgb 각각의 표준편차
std_r = imgs[:,0,:,:].std()
std_g = imgs[:,1,:,:].std()
std_b = imgs[:,2,:,:].std()
print(std_r,std_g,std_b)
```
