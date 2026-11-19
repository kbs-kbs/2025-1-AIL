# Let There Be Color (흑백 이미지 채색)

- 흑백 이미지를 컬러 이미지로 복원(채색)하는 모델
- 단순 CNN으로도 채색 모델은 만들 수 있지만, CNN만으로는 이미지 전체의 문맥(global context)을 효율적으로 활용하기 어려움
- → 지역 정보(local)와 전역 정보(global)를 동시에 학습하고, **이미지 분류기 앞단의 연산을 병행**해 전역 정보의 품질을 높임

> [!note] 채색 모델인데 왜 이미지 분류의 앞단 연산을 쓸까?
> 사물이 무엇인지(문맥)를 이해하면 더 자연스러운 색을 예측할 수 있기 때문.
> 하늘/바다, 사과/토마토는 모양이 비슷해도 색이 다름 → 분류기의 전역 특징 추출 구조를 빌려와 색 복원에 사용하고, 분류 결과 자체는 출력하지 않음

## 색공간(Color Space)

색을 수치로 표현하기 위한 체계. 컴퓨터는 색을 숫자로 처리하므로 용도에 따라 여러 종류를 씀.

| 색공간 | 용도 | 특징 |
|---|---|---|
| RGB | 화면 출력용 | R/G/B 조합. 밝기와 색상 정보가 명확히 분리되지 않음. 값을 각각 바꿀 때 사람이 어떤 색인지 직관적으로 예측하기 어려움 |
| CMYK | 인쇄용 | Cyan, Magenta, Yellow, Black |
| HSV / HSL | 색조 기반 표현 | 색조(Hue) 중심 |
| Lab | 인간 시각 기준 | L\*: 밝기(0=검정, 100=흰색), a\*: 녹색↔빨강, b\*: 파랑↔노랑 |

- **Lab의 장점**
  - 시각적으로 균등한 색공간
  - 밝기(L)와 색상(a, b)을 분리해서 조작할 수 있음
  - → 색 비교, 색 보정, **채색 모델**에 적합

## 모델 구조

- 입력: 흑백 이미지의 **L 채널**
- 출력: 컬러 이미지의 **a, b 채널**
- 구성 요소: Low-Level Feature Network / Mid-Level Feature Network / Global Feature Network / Fusion Layer / Colorization Network

```
        ┌─ Low-Level 특징 추출 ─┬─ Mid-Level 특징 추출 ─┐
입력 ───┤                       │                        ├─ 특징 합치기 ─ 컬러라이제이션 네트워크 ─ 출력
        └─ Global 특징 추출 ────┴────────────────────────┘
```

| 구성 | 역할 | 구조 |
|---|---|---|
| Low-Level Feature Network | 엣지, 질감 등 기초 시각 정보 추출 (작은 크기 특징) | 합성곱 레이어 여러 개 + 활성화 함수, stride로 해상도 축소 / 채널 수 증가 |
| Mid-Level Feature Network | 지역의 의미 정보 추출, 세부 구조 파악 (중간 크기 특징) | **풀링 없이** 합성곱만 사용해 해상도를 유지하며 특징 추출 (병합 전까지 지역 정보 보존) |
| Global Feature Network | 이미지 전체의 문맥 정보 파악 (전체 맥락) | 다운샘플링으로 전역 특징 추출, 이미지 분류기 앞단 구조를 차용 |
| Fusion Layer | 전역 + 지역 정보 결합 | 전역 특징을 지역 특징맵에 **broadcast** 하여 채널 방향으로 concat |
| Colorization Network | 원래 크기로 복원 후 픽셀 단위 a, b 채널 예측 | 업샘플링(ConvTranspose2d) + 합성곱 |

> [!note] broadcast
> 낮은 차원의 벡터를 공간적으로 확장하는 연산.
> Global 특징 `(B, 512, 8, 8)`을 지역 특징맵 크기 `(B, 256, 8, 8)`에 맞춰 `repeat` + `reshape` 후 채널 방향으로 `cat`.

---

# 실습: 채색 모델 만들기

## 1) 데이터 준비

- Kaggle `adityajn105/flickr8k` 데이터셋 사용 (자연 이미지 + 흑백 변환용)
- 이미지 256×256으로 리사이즈 후 **RGB → Lab 변환**
- 입력 `L` 채널은 흑백 이미지, 정답 `a, b` 채널은 색상 정보 → `return lab[0], lab[1:]`

```python
import cv2
import numpy as np
from torch.utils.data.dataset import Dataset

# RGB를 LAB으로 변환
def rgb2lab(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)

# LAB를 RGB로 변환 (이미지 출력용)
def lab2rgb(lab):
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

class AutoColoring(Dataset):
    def __init__(self):
        self.data = glob.glob("/kaggle/input/flickr8k/Images/*.jpg")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        rgb = np.array(Image.open(self.data[i]).resize((256, 256)))
        lab = rgb2lab(rgb)                      # RGB → LAB
        lab = lab.transpose((2, 0, 1)).astype(np.float32)  # (H, W, C) → (C, H, W)
        return lab[0], lab[1:]                  # 입력: L 채널 / 정답: a, b 채널
```

## 2) Low-Level Feature Extractor

- 흑백 이미지를 입력받아 엣지·질감 등 **저수준 특징** 추출
- Conv(3×3, stride 1·2) → BatchNorm → Sigmoid 를 7번 반복, 채널 64 → 512

```python
class LowLevel(nn.Module):
    def __init__(self):
        super(LowLevel, self).__init__()
        self.low1 = nn.Conv2d(1, 64, 3, 1, 1)
        self.lb1  = nn.BatchNorm2d(64)
        self.low2 = nn.Conv2d(64, 64, 3, 2, 1)     # stride 2 → 해상도 절반
        self.lb2  = nn.BatchNorm2d(64)
        self.low3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.lb3  = nn.BatchNorm2d(128)
        self.low4 = nn.Conv2d(128, 128, 3, 2, 1)
        self.lb4  = nn.BatchNorm2d(128)
        self.low5 = nn.Conv2d(128, 256, 3, 1, 1)
        self.lb5  = nn.BatchNorm2d(256)
        self.low6 = nn.Conv2d(256, 256, 3, 1, 1)
        self.lb6  = nn.BatchNorm2d(256)
        self.low7 = nn.Conv2d(256, 512, 3, 2, 1)
        self.lb7  = nn.BatchNorm2d(512)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        low = self.low1(x); low = self.lb1(low); low = self.sigmoid(low)
        ...                                          # low2 ~ low7 동일 반복
        return low
```

## 3) Mid-Level Feature Extractor

- 해상도는 그대로 두고(풀링 없음) 합성곱으로 **중간 수준(지역 의미)** 특징 추출
- 입력 채널 512, 출력 채널 256

```python
class MidLevel(nn.Module):
    def __init__(self):
        super(MidLevel, self).__init__()
        self.mid1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.mb1  = nn.BatchNorm2d(512)
        self.mid2 = nn.Conv2d(512, 256, 3, 1, 1)
        self.mb2  = nn.BatchNorm2d(256)
        self.sigmoid = nn.Sigmoid()
    # forward: conv → BN → Sigmoid 두 번
```

## 4) Global Feature Extractor

- Low-Level의 출력을 입력받아 다운샘플링하며 전역 특징 추출
- 마지막에 **MLP**(flatten → Linear)로 1차원 벡터화 → 분류기 앞단 구조 차용

```python
class GlobalLevel(nn.Module):
    def __init__(self):
        super(GlobalLevel, self).__init__()
        self.glob1 = nn.Conv2d(512, 512, 3, 2, 1)
        self.gb1   = nn.BatchNorm2d(512)
        self.glob2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.gb2   = nn.BatchNorm2d(512)
        self.glob3 = nn.Conv2d(512, 512, 3, 2, 1)
        self.gb3   = nn.BatchNorm2d(512)
        self.glob4 = nn.Conv2d(512, 512, 3, 1, 1)
        self.gb4   = nn.BatchNorm2d(512)
        self.glob5 = nn.Conv2d(512, 512, 3, 1, 1)
        self.gb5   = nn.BatchNorm2d(512)
        # 전역 특징 추출용 MLP (분류 task 아님, 전역 정보만 뽑음)
        self.fc1 = nn.Linear(in_features=32768, out_features=1024)  # 8*8*512
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=256)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        ...
        glo = torch.flatten(glo, start_dim=1)
        glo = self.fc1(glo); glo = self.sigmoid(glo)
        glo = self.fc2(glo); glo = self.sigmoid(glo)
        glo = self.fc3(glo); glo = self.sigmoid(glo)
        return glo
```

## 5) Colorization Network (디코더)

- 작업: 256×256×256 특징맵을 **a, b 채널(2채널)** 색상 맵으로 복원
- ConvTranspose2d로 업샘플링 (conv3 1배 → conv2·4 2배 → conv5 2배)

```python
class Colorization(nn.Module):
    def __init__(self):
        super(Colorization, self).__init__()
        self.color1 = nn.ConvTranspose2d(256, 128, 3, 1, 1)
        self.cb1 = nn.BatchNorm2d(128)
        self.color2 = nn.ConvTranspose2d(128, 64, 2, 2)   # UpSampling (64→64)
        self.cb2 = nn.BatchNorm2d(64)
        self.color3 = nn.ConvTranspose2d(64, 64, 3, 1, 1)
        self.cb3 = nn.BatchNorm2d(64)
        self.color4 = nn.ConvTranspose2d(64, 32, 2, 2)    # UpSampling (128→128)
        self.cb4 = nn.BatchNorm2d(32)
        self.color5 = nn.ConvTranspose2d(32, 2, 2, 2)     # UpSampling (256→256), 출력 2채널
        self.sigmoid = nn.Sigmoid()
```

## 6) 전체 모델 (AutoColoringModel)

```python
class AutoColoringModel(nn.Module):
    def __init__(self):
        super(AutoColoringModel, self).__init__()
        self.low  = LowLevel()
        self.mid  = MidLevel()
        self.glo  = GlobalLevel()
        self.fusion = nn.Conv2d(512, 256, 3, 1, 1)   # 전역(256) + 지역(256) = 512 → 256
        self.color  = Colorization()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        low = self.low(x)
        mid = self.mid(low)
        glo = self.glo(low)

        # 전역 특징을 지역 특징맵 크기에 맞게 broadcast
        fusion = glo.repeat(1, mid.shape[2] * mid.shape[3])
        fusion = torch.reshape(fusion, (1, 256, mid.shape[2], mid.shape[3]))

        # 채널 방향으로 결합: 256 + 256 = 512
        fusion = torch.cat([mid, fusion], dim=1)

        fusion = self.fusion(fusion)   # 512 → 256
        fusion = self.sigmoid(fusion)
        color = self.color(fusion)     # (B, 2, 256, 256)
        return color
```

## 7) 학습

- 손실: 예측한 a, b 채널과 정답 a, b 채널의 **MSE**
- L 채널은 입력으로만 쓰고, 손실은 색상 채널에서만 계산

```python
model = AutoColoringModel().to(device)
dataset = AutoColoring()
loader = DataLoader(dataset, batch_size=32, shuffle=True)
optim = Adam(params=model.parameters(), lr=0.01)

for epoch in range(1):
    iterator = tqdm.tqdm(loader)
    for L, AB in iterator:
        L = torch.unsqueeze(L, dim=1).to(device)   # (batch, 1, 256, 256)
        optim.zero_grad()
        pred = model(L)                            # (batch, 2, 256, 256)
        loss = nn.MSELoss()(pred, AB.to(device))
        loss.backward()
        optim.step()
        iterator.set_description(f"epoch {epoch} loss: {loss.item()}")

torch.save(model.state_dict(), "AutoColor.pth")
```

## 8) 결과 확인

- 예측 a, b와 입력 L을 다시 합쳐 Lab → RGB 변환 후 원본과 비교

```python
with torch.no_grad():
    model.load_state_dict(torch.load("AutoColor.pth", map_location=device))
    pred_AB = model(torch.unsqueeze(test_L, dim=0).to(device))

real_img = np.concatenate((test_L, test_AB), axis=0)         # (3, H, W)
real_img = lab2rgb(real_img.transpose(1, 2, 0).astype(np.uint8))

pred = torch.cat([test_L, pred_AB], dim=0).squeeze(0).numpy()
pred_img = lab2rgb(pred.transpose(1, 2, 0).astype(np.uint8))
```

## 생각해보기

1. **왜 RGB 대신 Lab을 쓰나?** → 채색 문제에서 입력(밝기)과 출력(색상)을 분리할 수 있어서. L 채널 하나를 입력으로 주고 a, b만 예측하면 되므로 문제가 단순해짐
2. **Global 특징을 왜 broadcast 하나?** → 전역 특징은 1차원(256,)인데 디코더는 공간 정보가 필요하므로, 모든 위치에 같은 전역 정보를 복제해 넣어 "이 픽셀이 어떤 물체의 일부인지"를 참고하게 함
3. **분류 손실이 없는데 분류기 구조를 쓰는 이유?** → 분류 결과를 출력하는 게 아니라, 문맥을 잘 요약하는 특징 추출기로만 사용
