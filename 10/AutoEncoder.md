# 오토인코더 (AutoEncoder)

- 입력 데이터를 압축해 **latent vector** 로 요약하고, 이를 바탕으로 입력을 복원하는 **비지도 학습** 신경망
- 흐름: 이미지 입력 → 인코더 → **잠재 벡터(latent vector)** → 디코더 → 복원된 이미지
- 출력은 입력과 가능한 한 유사해야 함
- 손실 함수는 주로 **MSE(평균 제곱 오차)** 사용

> [!note] 지도학습이 아님
> 정답 레이블을 쓰는 것이 아니라, **입력을 그대로 복원**하는 것이 목적이므로 비지도 학습으로 분류

## 왜 AutoEncoder를 쓰는가?

- 정답을 맞추는 것도, 입력을 그대로 복사하는 것도 아님
- **데이터의 중요한 특성을 잘 요약하는 인코더**와, 그 정보로 이미지를 잘 복원하는 **디코더**를 만드는 것이 목적
- 학습 과정에서 모델은 입력의 구조적/의미적 핵심을 latent vector에 응축하고, **그 요약만으로** 원본에 가깝게 복원하도록 학습됨
- 즉, 정확한 복사(X) → **의미 있는 요약 후 의미 있는 복원(O)**
  - 예) 사람 얼굴 사진에서 주요 특징만 잘 뽑아내어, 그 특징만으로 입력 얼굴과 최대한 비슷한 이미지를 복원

---

## 구조

```
[Encoder] → (latent vector) → [Decoder]
```

| 구성 | 역할 | 사용 레이어 |
|---|---|---|
| Encoder | 입력을 점점 압축해 중요한 특징만 남김 | Conv2D 또는 Dense |
| Latent Vector | 입력의 핵심 정보만 담은 압축된 표현 (입력의 요약) | - |
| Decoder | Latent vector로부터 입력과 비슷하게 복원 | ConvTranspose2D 또는 Dense |

- 인코더와 디코더는 주로 **대칭 구조**
- 출력은 입력과 **같은 크기와 형태**

### Latent Vector

- "Latent" = 잠재적인, 잠복해 있는
- 추상적인 정보만 포함하며 이 벡터를 기반으로 복원을 수행 → **원본 데이터의 아주 잘 정리된 요약본**
- 형태는 1D vector `[128]` 일 수도 있고, 이미지 데이터를 다루는 AutoEncoder에서는 작은 3D tensor `[8×8×64]` 일 수도 있음
  - 원본 이미지의 요약 정보만 남긴 표현이라는 점에서 이것도 latent vector라고 부름

> [!note] feature map과 latent vector는 본질적으로 다른가?
> - **U-Net의 feature map**: 디코더로 전달되는 **중간 표현**. skip connection으로 여러 feature map이 함께 디코더에서 사용됨
> - **AutoEncoder의 latent vector**: 전체 정보를 요약한 **핵심 표현**. 디코더는 **오직 이 벡터 하나**로 복원을 시도함
> → 형태는 비슷할 수 있지만, latent vector가 훨씬 압축되어 있고 추상적인 정보를 담고 있음

### Convolutional AutoEncoder (CAE)

- 이미지 데이터에 적합한 합성곱 기반 AutoEncoder
- Dense 대신 Conv2D, MaxPooling 등 CNN 연산을 사용해 이미지의 **공간적 구조(위치, 형태)** 를 보존
- 인코더: `Conv2D → ReLU → MaxPooling` 반복
- 디코더: `ConvTranspose2D → ReLU → UpSampling2D` 반복
- 장점: 공간 정보 유지, 이미지 데이터에 특화

---

## 활용 분야

| 활용 | 설명 |
|---|---|
| Denoising AutoEncoder | 원본 이미지 → [노이즈 추가] → 인코더 → latent → 디코더 → 깨끗한 이미지 |
| Anomaly Detection | 정상 데이터로만 학습하면 정상 입력은 잘 복원하지만, 학습과 다른 **이상 데이터는 복원 오차가 커짐** → 오차가 크면 이상으로 판단 |
| Dimensionality Reduction | latent vector를 차원 축소 결과로 사용 (PCA처럼) |

---

## AutoEncoder vs. U-Net

둘 다 encoder-decoder 구조지만 **복원의 목적과 방식**이 다름.

- **U-Net**: skip connection으로 인코딩 중 나온 원본에 가까운 feature map들을 디코딩 단계에서 받아 **정밀한 복원**에 사용
- **AutoEncoder**: 요약한 **latent vector 하나만**으로 원본과 비슷하게 복원

| 항목 | U-Net | Autoencoder |
|---|---|---|
| 목적 | 픽셀 단위 예측 (분할 등) | 입력을 그대로 복원 |
| 입력-출력 관계 | 입력과 동일한 해상도에서 목적에 맞는 값을 얻음 | 입력과 동일한 출력을 얻음 |
| Skip connection | O (복원할 때 힌트가 됨) | X |
| 출력 특징 | 경계를 정밀하게 유지한 이미지 | latent vector만 기반으로 복원한 이미지 |
| 압축 강도 | 낮음 (정보를 많이 유지) | 높음 (latent vector로 압축) |
| 사용 분야 | 의료 영상, 이미지 분할 | 노이즈 제거, 이상 탐지 |

> 정밀한 위치 정보나 경계 보존이 중요한 경우(예: 의료 영상)는 skip connection을 쓰는 **U-Net이 더 유리하고 성능이 높음**
> (참고: u-net에서 skip-connection이 없는 형태가 오토인코더)

---

# 실습: 이미지 노이즈 제거 (Denoising AutoEncoder)

1. **데이터 준비** — 원본 이미지 데이터셋을 불러와 텐서로 변환
2. **노이즈 추가** — 입력에 임의의 노이즈를 더한 이미지를 모델 입력으로 사용 (정답은 **깨끗한 원본**)
3. **모델 정의** — 인코더(Conv + Pooling) → latent vector → 디코더(ConvTranspose + UpSampling)의 대칭 구조
4. **학습** — 입력(노이즈 이미지)과 출력(복원 이미지) 사이의 MSE를 최소화
   - 모델이 "노이즈는 버리고 원본의 구조만 요약해 복원하는" 방법을 배우게 됨
5. **결과 확인** — 노이즈 이미지 / 복원 이미지 / 원본을 나란히 비교

> [!note] Denoising AutoEncoder의 원리
> 입력에 노이즈를 섞어도 **정답은 깨끗한 이미지**이므로, 모델은 latent vector에 노이즈가 아닌 **원본의 핵심 구조만** 담도록 강제됨
> → 결과적으로 노이즈에 강건한(robust) 표현을 학습함

참고: https://wikidocs.net/193827
