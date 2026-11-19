
# U-Net

## 등장 배경

| 문제 | 입력 | 출력 |
|---|---|---|
| Image Classification | 이미지 | 이미지가 속한 class |
| Image Segmentation | 이미지 | segmentation map (픽셀별 class) |

- 일반 CNN: `이미지 → Convolution → Pooling → Fully connected → Class` → 분류는 잘하지만 **위치 정보가 사라짐**
- 단순 FCN(Fully Convolutional Network): `이미지 → Convolution → Pooling → Convolution → Segmentation map`
  - 기존 위치 정보 소실로 모서리·경계선이 뭉개진 저품질 결과
- **(IDEA) 이미지의 위치 정보를 유지해줄 skip connection이 필요하다!**

## U-Net의 특징

- Encoder-Decoder 구조와 **Skip connection**을 통해 **공간 정보와 추상 정보를 동시에** 활용
- 전체 구조가 대칭적인 **U자 형태**
- Encoder-Decoder 구조로 해상도 감소-복원 흐름을 따름
- FCN과 비교하면 **소규모 데이터셋에서도 높은 정확도** → 의료 영상처럼 데이터 수집이 어려운 분야에서 많이 활용

## 관련 개념

- **Encoder**: Convolution과 Pooling을 반복하며 이미지를 점점 압축하고 추상적 특징을 추출
- **Decoder**: Upsampling을 통해 추출된 특징(압축된 정보)을 점진적으로 복원하고 원래 크기와 유사하게 재구성
- **Upsampling**: 추출한 특징을 이용해 이미지를 복원하는 과정, 해상도를 키우는 과정
- **Downsampling**: 해상도(H×W)를 줄이는 것. U-Net에서는 인코더 내부의 해상도 축소 연산(Max Pooling, Strided Convolution)을 가리킴
  - (참고) ResNet의 다운샘플은 `F(x) + x` 를 위해 해상도를 줄이면서 **채널 수를 늘려 모양을 맞추는** 작업이었음
- **Skip Connection**: 같은 깊이의 인코더 출력과 디코더 입력을 직접 연결
  - 두 feature map을 **concatenate**(채널 추가)하거나 값을 더함
  - 인코더 초반의 공간 정보를 디코더에서 고수준 의미 정보와 함께 사용해 **경계면 정보 손실을 완화**

## U-Net의 구조

- Encoder와 Decoder가 각각 여러 층으로 구성되며, 같은 깊이의 인코더-디코더 층 사이에 skip connection이 있음
- Encoder에서 추출한 feature는 decoder에서 고해상도 출력을 복원할 때 직접 결합되어 사용됨
- 마지막에는 **1×1 convolution** 으로 클래스 수만큼 채널을 줄이고, 각 픽셀마다 분류 결과를 출력

## 활용 분야

- Image Segmentation, Image Restoration 등

---

# (참고) U-Net 아키텍처 정리

U-Net은 주로 이미지 분할(Image Segmentation)에 사용되는 합성곱 신경망(CNN) 기반의 아키텍처로, 2015년 의료 영상 분석 분야에서 처음 제안되었습니다[3][6]. 이름처럼 전체 구조가 U자 형태를 띠며, 인코더(Encoder, 축소 경로)와 디코더(Decoder, 확장 경로)로 대칭적으로 구성되어 있습니다[1][3][5].

---

**아키텍처 구조**

- 인코더(Contracting Path):  
  입력 이미지를 점차적으로 축소하며(다운샘플링) 주요 특징(feature)을 추출합니다. 각 단계는 2개의 3x3 컨볼루션과 ReLU 활성화, 그리고 2x2 맥스풀링(Max Pooling)으로 이루어져 있습니다. 이 과정에서 공간 해상도는 줄어들고 채널 수는 늘어납니다[1][3][7].

- 디코더(Expanding Path):  
  추출된 특징을 바탕으로 이미지를 점차적으로 복원(업샘플링)하여 원래 해상도에 가까운 분할 결과를 만듭니다. 업샘플링(Transposed Convolution 또는 업컨볼루션)과 2개의 3x3 컨볼루션, ReLU 활성화로 구성됩니다[1][7].

- 스킵 연결(Skip Connection):  
  인코더의 각 단계에서 얻은 고해상도 특징 맵을 디코더의 대응 단계에 직접 연결(concatenation)합니다. 이를 통해 디코더가 위치 정보와 세밀한 경계 정보를 잘 복원할 수 있습니다[1][2][4].

- 브릿지(Bridge):  
  인코더와 디코더의 중간에 위치하는 병목(bottleneck) 부분으로, 가장 압축된 특징을 담고 있습니다[1][5].

---

**U-Net의 특징 및 장점**

- U자형 대칭 구조로 인해 고해상도 정보와 저해상도 정보가 모두 활용되어, 미세한 경계까지 정밀하게 분할할 수 있습니다[3].
- 스킵 연결 덕분에 위치 정보 손실을 최소화하면서도, 깊은 신경망의 장점(복잡한 특징 추출)을 모두 누릴 수 있습니다[1][2][4].
- 적은 양의 주석 데이터로도 효과적으로 학습할 수 있어, 특히 의료 영상 등 데이터가 귀한 분야에서 강점을 보입니다[6].
- 완전 연결 계층 없이, 입력 이미지의 유효한 부분만을 사용해 픽셀 단위의 분할 맵을 생성합니다[6].

---

**적용 분야**

- 의료 영상(병변, 세포, 장기 분할 등)
- 위성 이미지 분석, 생물학적 이미지, 일반 컴퓨터 비전의 다양한 세분화 문제 등[3][6].

---

**정리**

U-Net은 인코더-디코더 구조와 스킵 연결을 결합한 U자형 아키텍처로, 이미지의 전역적 맥락과 세부 위치 정보를 동시에 활용해 높은 정확도의 이미지 분할을 달성하는 것이 핵심입니다[1][3][6].

Citations:
[1] https://velog.io/@lighthouse97/UNet%EC%9D%98-%EC%9D%B4%ED%95%B4
[2] https://pasus.tistory.com/204
[3] https://ai-bt.tistory.com/entry/U-Net-%EC%9D%98-%EC%9D%B4%ED%95%B4
[4] https://wikidocs.net/148870
[5] https://jaylala.tistory.com/entry/%EB%94%A5%EB%9F%AC%EB%8B%9D-with-%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0U-Net%EC%9D%B4%EB%9E%80-U-Net-Convolutional-Networks-for-Biomedical-Image-Segmentation
[6] https://wikidocs.net/128392
[7] https://computing-jhson.tistory.com/61
[8] https://joungheekim.github.io/2020/09/28/paper-review/

---
Perplexity로부터의 답변: https://www.perplexity.ai/search/u-net-akitegceoe-daehae-seolmy-MDwlldCnSCKYnGuWao__RQ?utm_source=copy_output

---

# 실습: 동물 사진과 배경 분리하기 (Oxford-IIIT Pet)

- 데이터셋: `https://www.robots.ox.ac.uk/~vgg/data/pets/` (반려동물 사진 + 픽셀 단위 마스크)
- 흐름: 데이터 살펴보기 → 데이터 전처리(이미지·마스크 동시 리사이즈, 텐서 변환) → U-Net 모델 정의 → 학습 → 결과 확인
- 손실 함수: **BCEWithLogitsLoss** (배경/전경 2클래스 분할)
  - 모델 마지막에 Sigmoid를 두지 않고 로짓을 그대로 출력 → 수치적으로 안정적인 이진 교차엔트로피
  - (참고) BCEWithLogitsLoss = Sigmoid + BCELoss 를 하나로 합친 것
- 1 에폭만 학습해도 동물 영역이 분리되는 결과를 확인할 수 있음

## 생각해보기

1. **Encoder-Decoder 구조란 무엇인가?** 입력을 압축해 특징을 뽑는 인코더와, 그 특징으로 원래 크기의 출력을 복원하는 디코더로 이루어진 구조
2. **U-Net의 Encoder 층이 10층이라면 Decoder 층은 몇 층이어야 하는가?** 대칭 구조이므로 10층
3. **Skip Connection의 역할은?** 같은 깊이의 인코더 feature map을 디코더에 직접 전달해, 다운샘플링으로 잃어버린 **위치·경계 정보**를 복원 단계에서 되살리는 것
4. **BCEWithLogitsLoss** — Sigmoid를 모델 밖으로 빼고 로짓에 바로 적용하는 이진 교차엔트로피 (log-sum-exp 트릭으로 계산이 안정적)
