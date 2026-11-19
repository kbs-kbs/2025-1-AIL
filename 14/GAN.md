# GAN (Generative Adversarial Network)

- Generative(생성의) + Adversarial(적대적인) + Network(신경망)
- 새로운 데이터를 생성하는 딥러닝 모델 (2014, Ian Goodfellow)
- **두 신경망이 경쟁하며 학습**하는 것이 GAN만의 특징

| 구성 | 역할 |
|---|---|
| Generator (생성자) | 가짜 데이터를 생성. 학습을 통해 점점 더 정교한 가짜를 만듦 |
| Discriminator (판별자) | 데이터가 진짜인지 가짜인지 판별. 점점 더 정확하게 진짜/가짜를 구별 |

## 전체 흐름

```
[무작위 노이즈 z] → Generator → 가짜 이미지 ─┬→ Discriminator → 진짜/가짜 판별
[실제 이미지]  ────────────────────────────┘
```

- 생성자는 판별자를 **속이려고** 하고, 판별자는 **속지 않으려고** 함 → 서로 적대적으로 발전
- 학습이 끝나면 생성자는 실제와 구분하기 어려운 데이터를 만듦

---

## Generator 내부 구조 (예: 64×64 얼굴 생성)

무작위 노이즈로부터 이미지를 생성. 5번 업샘플링(1 → 4 → 8 → 16 → 32 → 64)으로 해상도 확대.

| 순서 | 계층 | 출력 채널 | 입력 → 출력 |
|---|---|---|---|
| 1 | ConvTranspose2d | 512 | (1, 100, 1, 1) → (1, 512, 4, 4) |
| 2 | BatchNorm2d | 512 | 유지 |
| 3 | ReLU | 512 | 유지 |
| 4 | ConvTranspose2d | 256 | (1, 512, 4, 4) → (1, 256, 8, 8) |
| 5~6 | BatchNorm2d + ReLU | 256 | 유지 |
| 7~9 | ConvTranspose2d + BN + ReLU | 128 | → (1, 128, 16, 16) |
| 10~12 | ConvTranspose2d + BN + ReLU | 64 | → (1, 64, 32, 32) |
| 13 | ConvTranspose2d | 3 | (1, 64, 32, 32) → (1, 3, 64, 64) |
| 14 | Tanh | 3 | 픽셀값 -1~1 정규화 |

## Discriminator 내부 구조

이미지를 입력받아 진짜/가짜를 판별. 진짜 레이블 1, 가짜 레이블 0인 **이진 분류**로 학습.
5번 다운샘플링(64 → 32 → 16 → 8 → 4 → 1).

| 순서 | 계층 | 출력 채널 | 입력 → 출력 |
|---|---|---|---|
| 1 | Conv2d | 64 | (1, 3, 64, 64) → (1, 64, 32, 32) |
| 2~3 | BatchNorm2d + LeakyReLU | 64 | 유지 |
| 4~6 | Conv2d + BN + LeakyReLU | 128 | → (1, 128, 16, 16) |
| 7~9 | Conv2d + BN + LeakyReLU | 256 | → (1, 256, 8, 8) |
| 10~12 | Conv2d + BN + LeakyReLU | 512 | → (1, 512, 4, 4) |
| 13 | Conv2d | 1 | (1, 512, 4, 4) → (1, 1, 1, 1) |
| 14 | Sigmoid | 1 | 진짜일 확률(0~1) 반환 |

## 활성화 함수 정리

| 함수 | 위치 | 출력 범위 | 사용 이유 |
|---|---|---|---|
| ReLU | Generator 은닉층 | 0 ~ ∞ | 음수 제거, 빠른 수렴, 비선형성 확보 |
| Tanh | Generator 마지막 | -1 ~ 1 | 이미지 픽셀값 범위에 맞춤 |
| LeakyReLU | Discriminator 은닉층 | 음수 영역 포함 | 음수도 일부 반영 → **죽은 뉴런 문제** 방지, 안정적 학습 |
| Sigmoid | Discriminator 마지막 | 0 ~ 1 | 이진 분류를 위한 확률값 |

---

## GAN의 활용

- 무작위 입력으로 **사람 얼굴 생성**
- **저해상도 → 고해상도** 이미지 생성 (SRGAN)
- 텍스트 기반 이미지 생성
- 이미지 **스타일 변환**
- 의료 영상 노이즈 제거
- 디자인 시안 제작

---

# 실습 1: 사람 얼굴을 생성하는 GAN

### 데이터 준비
- CelebA: `https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html`
- In-The-Wild Images → `Img/img_align_celeba.zip` 을 내 드라이브에 바로가기 추가

### 모델 정의 (DCGAN 스타일)

```python
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(100, 512, kernel_size=4, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.gen(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 1, kernel_size=4),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)
```

### 가중치 초기화

```python
def weights_init(m):
    # 층의 종류 추출
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # 합성곱 초기화 (평균 0, 표준편차 0.02의 정규분포)
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        # 배치 정규화 초기화
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

G = Generator().to(device)
G.apply(weights_init)
D = Discriminator().to(device)
D.apply(weights_init)

G_optim = Adam(G.parameters(), lr=0.0001, betas=(0.5, 0.999))
D_optim = Adam(D.parameters(), lr=0.0001, betas=(0.5, 0.999))
```

> [!note] betas=(0.5, 0.999)
> DCGAN 논문에서 제안한 설정. 모멘텀 항을 낮춰 GAN 학습을 안정화함.

### 학습 루프 (핵심)

```python
for epochs in range(1):
    for i, data in enumerate(loader, 0):
        # ------------------ 판별자 학습 ------------------
        D_optim.zero_grad()
        label = torch.ones_like(data[1], dtype=torch.float32).to(device)   # 실제 이미지 = 1
        label_fake = torch.zeros_like(data[1], dtype=torch.float32).to(device)  # 가짜 = 0

        # 실제 이미지를 판별자에 입력
        real = D(data[0].to(device))
        Dloss_real = nn.BCELoss()(torch.squeeze(real), label)
        Dloss_real.backward()

        # 가짜 이미지 생성 후 판별
        noise = torch.randn(label.shape[0], 100, 1, 1).to(device)
        fake = G(noise)
        output = D(fake.detach())     # 생성자로 역전파되지 않도록 detach
        Dloss_fake = nn.BCELoss()(torch.squeeze(output), label_fake)
        Dloss_fake.backward()

        Dloss = Dloss_real + Dloss_fake
        D_optim.step()

        # ------------------ 생성자 학습 ------------------
        # (판별자가 가짜를 '진짜(1)'로 판단하도록 속이는 방향으로 학습)
        # G_optim.zero_grad()
        # output = D(fake)
        # Gloss = nn.BCELoss()(torch.squeeze(output), label)
        # Gloss.backward(); G_optim.step()
```

- 판별자: 실제는 1, 가짜는 0으로 학습
- 생성자: 판별자가 가짜를 **1(진짜)** 로 판단하도록 학습 → 판별자를 속이는 방향
- `fake.detach()`: 판별자 학습 시 생성자까지 기울기가 흘러가지 않도록 분리

---

# 실습 2: 화질을 개선하는 SRGAN

- 저해상도 이미지를 입력받아 **고해상도 이미지**를 생성하는 GAN
- 생성자: 저해상도 → 고해상도 업샘플링 / 판별자: 고해상도 이미지가 진짜인지 생성된 것인지 판별
- 1 에폭만 학습해도 생성 결과를 눈으로 확인할 수 있음 (학습이 오래 걸리는 모델이라 소규모로 실습)

참고: https://medium.com/data-science/deep-learning-introduction-of-generative-adversarial-networks-gans-ae22c4350b1f
