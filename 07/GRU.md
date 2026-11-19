# GRU

## GRU의 등장 배경

LSTM은 RNN의 장기 의존성 문제를 해결하였지만, 구조가 복잡하여 학습 시간이 오래 걸린다는 단점이 있음
그래서 LSTM의 성능은 유지하면서도 더 간단하고 학습이 빠른 구조가 필요해짐 ⇒ GRU(Gated Recurrent Unit)의 등장

- 3(또는 4)개의 gate가 있는 LSTM과 달리, GRU는 **단 2개의 gate**만 존재함
- GRU의 두 gate들이 정보의 흐름을 정교하게 조절함
  - **Update Gate**: 기억을 얼마나 업데이트할까? (과거 정보와 새 정보의 비율 조절)
  - **Reset Gate**: 이전 기억을 얼마나 지워버릴까? (과거 정보를 얼마나 적게 쓸지 조절)
- 2개의 state(hidden state, cell state)를 사용하던 LSTM과 달리, GRU는 **hidden state만** 사용함
- reset gate와 update gate는 동시에 계산될 수 있으며, 서로에게 영향을 주지 않음.
  - 오직 `g_t` 계산을 할 때에만 reset gate의 연산 결과가 사용되고, 최종 hidden state인 `h_t` 계산에서만 update gate가 사용됨

### gate 비유

| 비유 | 게이트 |
|---|---|
| "이 문제는 전에 풀다 말았는데, 다시 보니까 방법이 많이 달라. 방법을 새로 생각해보자." | Reset Gate |
| "이번 설명이 너무 쏙쏙 들어왔어! 이거 위주로 기억하고, 전에 헷갈리던 건 덜 기억하자." | Update Gate |

### 관련 기호

| 기호 | 의미 |
|---|---|
| `h_(t-1)` | 이전 시점의 은닉 상태 (기존의 기억) |
| `x_t` | 현재 시점의 입력 |
| `z_t` | Update Gate (무엇을 기억에서 지우고 무엇을 유지할지 결정) |
| `r_t` | Reset Gate (기존 기억을 얼마나 무시할지 결정) |
| `g_t` | Candidate State (새로운 hidden state의 후보, 새로운 정보의 후보) |
| `h_t` | 최종 hidden state (새로 가지고 있을 기억) |
| `σ` | sigmoid 활성화 함수 (0~1) |
| `tanh` | hyperbolic tangent 활성화 함수 (-1~1) |

- `z_t`, `r_t`, `g_t` 모두 이전 상태 `h_(t-1)` 과 현재 입력 `x_t` 를 이용해 계산함
- `h_t` 는 update gate `z_t` 로 예전 기억과 새 정보를 혼합함
  - `z_t` 가 1에 가까우면 새 정보 `g_t` 를 더 많이 반영
  - `z_t` 가 0에 가까우면 과거 정보 `h_(t-1)` 을 더 많이 반영

> [!note] W의 첨자 규칙
> 첫 번째 첨자: 입력 소스 / 두 번째 첨자: 목적지 게이트
> 예) `W_xz` — 현재 입력 `x_t` 를 update gate 계산에 반영 / `W_hz` — 이전 hidden state를 update gate에 반영

# CRNN

```
Input -> CNN Layer -> RNN Layer -> Output
```

- **CRNN = CNN + RNN**
  - CNN: 이미지에서 특징(feature) 추출에 탁월 (선, 모서리, 패턴)
  - RNN: 시간 순서가 있는 데이터(sequence) 처리에 유용 (번역, STT, 이미지 캡셔닝)
- **CNN Layer**: 2D 이미지를 여러 feature map으로 변환. 이때 CNN이 **시간축을 보존하도록** 커널을 설계해야 함
  (가로로 긴 글자 이미지 → CNN 출력도 글자 순서를 시퀀스로 해석할 수 있는 형태여야 함)
- **RNN Layer**: CNN의 출력(feature sequence)을 시간 순서대로 해석 (Vanilla RNN, GRU, LSTM 등)
- **Output Layer**: 최종 값(글자, 음소 등) 계산. 일반적으로 **CTC** 손실 함수 사용

# CTC Loss

- CTC(Connectionist Temporal Classification)라는 손실함수를 사용함
- 정답 시퀀스의 길이와 예측 시퀀스의 길이가 다를 때 정렬 없이 학습할 수 있는 손실 함수
  : 즉, 하나의 정답을 표현하는 방법이 여러 가지 존재할 때, 가능한 모든 표현 방식을 고려하여 손실을 계산하도록 설계된 함수

## 왜 필요한가?

- 일반적인 분류 문제에서는 예측[index]와 정답[index]가 1:1로 대응했음
- OCR·음성 인식처럼 **입력 길이는 길고 정답 길이는 짧은** 경우가 있음
  - 예) CNN 출력 시퀀스 길이가 180인데 정답은 `"dogs"`(4글자) → 어떤 프레임이 어떤 글자에 해당하는지 정렬 정보가 없음
- CTC Loss로 해결 가능

## 동작

```
[예측 시퀀스]     d   d   o   o   o   g   s   s
[중복 제거]       d   -   o   -   -   g   s   -   (반복 문자와 공백 제거)
[최종 예측]               d       o       g   s   → "dogs"
```

- `"a"` 를 나타내는 표현은 `aa`, `_a`, `a_` 처럼 여러 가지가 있음
- 정답이 `"dogs"` 라는 것은 알지만 어느 시점에서 각 글자를 예측했는지는 모르므로, **"dogs"로 해석될 수 있는 모든 시퀀스 경로**를 인정함
  - `d d o o o g s s`, `d d o o g g s s`, `_ d _ o o _ g s`, `d _ _ o g g _ s`, `d d d o o g g s s` ...
- CTC는 이 **모든 경로의 확률을 더한 값**을 정답 `"dogs"` 의 확률로 간주
- 모델은 이 확률(정답을 맞출 확률)이 최대한 높아지는 쪽으로 학습되고, 그 과정에서 계산되는 손실값이 CTC Loss

# 실습: CRNN을 활용한 캡차 텍스트 인식

- 목표: CAPTCHA 이미지 속 문자를 자동으로 인식하는 신경망 구현
- 네트워크 구조: **CNN(ResNet) + RNN(GRU)**
- 새로운 개념: **CTC Loss**
- 입력: 영어 소문자 및 숫자가 포함된 CAPTCHA 이미지 / 출력: 예측된 문자 시퀀스

# 추가 개념: Batch Normalization

- 모델에 입력으로 들어오는 **배치의 값 분포가 서로 다르면** 배치마다 출력값의 분포가 달라짐 → 학습에 악영향(학습이 불안정해짐)
- 이를 해결하기 위해 고안된 것이 Batch Normalization
- 즉, 배치 단위를 정규화하는 기법이며, 각 배치의 **출력 분포를 일정하게** 만들어 줌

> [!note] Image Normalization(입력 정규화)과의 차이
> - Image Normalization: **입력층**에서 입력 이미지의 픽셀 분포를 정규화 → 시각적으로 달라짐
> - Batch Normalization: 각 층의 **출력(활성값)** 을 배치 단위로 정규화 → 시각적으로는 달라지지 않지만 학습 안정화·과적합 방지 효과

참고: https://wikidocs.net/22889
