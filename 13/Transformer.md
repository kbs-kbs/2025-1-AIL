# Transformer

- 논문 **"Attention Is All You Need"**(2017)에서 소개된 딥러닝 모델
- 순차적 처리 없이도 입력 시퀀스의 관계를 효과적으로 학습하도록 설계됨

## Transformer의 주요 특징

- **Self-Attention Mechanism**: 입력 시퀀스 내의 모든 단어 간의 관계를 동적으로 학습
- **Encoder-Decoder 구조** 사용
- **병렬 처리**: 전체 시퀀스를 동시에 처리 → RNN(순차 처리)보다 학습 속도가 빠름
- 다양한 활용: 자연어 처리, 컴퓨터 비전(ViT), 음성 처리 등

> [!note] 디코더는 언제 필요한가?
> 디코더는 순차적인 **생성** 작업에 필요함.
> 단순 분류 목적(예: BERT)일 경우에는 인코더만 있으면 됨.

---

## 전체 구조

```
(입력 데이터) → 토크나이저 → 임베딩 레이어 → 인코더 N개 → 디코더 N개 → 출력층 → (출력 결과)
```

| 구성 | 역할 |
|---|---|
| 토크나이저(Tokenizer) | 입력 텍스트를 토큰화하고 정수 ID 시퀀스로 인코딩 |
| 임베딩 레이어(Embedding layer) | 정수 ID 시퀀스를 고차원 벡터(임베딩)로 변환하고 위치 정보를 더함 |
| Encoder | 전체 시퀀스를 받아 각 단어가 문장 내 다른 모든 단어와 어떤 관계를 맺는지 학습. 여러 층 쌓임 |
| Decoder | 이전에 생성한 단어들과 인코더의 출력을 입력받아 다음 단어를 순차적으로 예측. 여러 층 쌓임 |
| 출력층(Output Layer) | 문제 유형에 맞는 최종 결과 생성 |

### 인코더 스택 (n회 반복)

1. **Input Embedding + Positional Encoding** — 입력 단어를 임베딩하고 위치 정보를 더함
2. **Multi-Head Self-Attention** — 입력 문장 안의 단어들끼리 서로 어떤 관계인지 학습
3. **Feed Forward + Add & Norm** — 비선형 변환과 정규화

### 디코더 스택 (n회 반복)

1. **Output Embedding + Positional Encoding** — 이전에 생성된 출력 단어들을 임베딩 + 위치 정보
2. **Masked Multi-Head Self-Attention** — 디코더 내부의 Self-Attention. 현재 시점 이전 단어까지만 참조하도록 masking
3. **Linear + Softmax → Output Probabilities** — 단어장 크기 벡터로 변환 후 확률로 변환, 확률이 가장 높은 단어 출력

---

## 구성 요소 자세히

### Positional Encoding

토크나이저는 문장을 정수 ID로 바꾸고, 임베딩은 ID를 벡터로 바꿈. **여기에는 순서 정보가 없음** → 위치 인코딩을 더해줌

```text
"The cat sits." → ["the", "cat", "sit", "##s", "."] → [101, 1996, 4937, 3523, 2015, 1012, 102]

[101]  → [0.12, -0.38, 0.45, 0.88]     # 임베딩
"cat"  → [0.10, 0.40, -0.50, 0.30]     # 임베딩
PE[1]  → [0.00, 0.84, 0.91, 0.14]      # 위치 인코딩 (index=1)
```

### Self-Attention

- 입력 시퀀스 내의 모든 단어들끼리의 관계(의존성)를 동적으로 파악하는 메커니즘
- 단어마다 **문맥을 반영한 표현**을 만듦
- 전체 토큰 간의 관계를 **동시에** 계산 (병렬 계산)

> "The animal didn't cross the street because it was too tired."
> 단어 **it** 이 가리키는 것 → Self-Attention은 `it` 이 `street` 보다 **`animal` 과 더 높은 관련성**이 있다고 판단하도록 학습됨

### Multi-Head Self-Attention

- Self-Attention을 여러 방식으로 여러 번 실행해 **정보의 시야를 넓힘** (다양한 시각으로 문장을 바라봄)
  - head1: 문법적 관계 / head2: 의미적 유사성 / head3: 위치 정보 ...
- 서로 다른 가중치를 병렬적으로 계산함

### Feed Forward + Add & Norm

- **FFN**: Self-Attention 결과를 입력받아 두 선형 레이어 사이에 ReLU 등 활성화 함수 사용 → 비선형성 추가
- **Add**: FFN 출력과 원래 입력을 잔차 연결(Residual Connection) → 기울기 소실 방지, 원본 정보 유지
- **LayerNorm**: 잔차가 더해진 결과를 정규화 → 학습 안정화, 수렴 속도 개선

### Masked Multi-Head Self-Attention

- 아직 생성하지 않은 **미래의 단어를 참조하지 않도록** Attention Score를 마스킹(안 보이게)해서 계산
- 예측 시점에 미래의 정답 단어를 참조하면 부정행위가 되므로, 이후 단어에 대한 attention을 0으로 제한

```text
        t0   t1   t2
t0      O    X    X
t1      O    O    X
t2      O    O    O
```

### Linear + Softmax → Output

- **Linear**: 디코더 마지막 출력을 단어 집합(단어장) 크기의 벡터로 매핑
- **Softmax**: 각 단어가 나올 확률로 변환 → 가장 확률이 높은 단어 출력

---

## Transformer 기반 모델들

| 모델 | 인코더 | 디코더 | 사용 목적 |
|---|---|---|---|
| BERT (Bidirectional Encoder Representations from Transformers) | O | X | 분류 / 분석 |
| T5 (Text-to-Text Transfer Transformer) | O | O | 텍스트 생성 / 번역 / 요약 |
| BART (Bidirectional and Auto-Regressive Transformer) | O | O | 텍스트 생성 / 번역 / 요약 |
| GPT (Generative Pre-trained Transformer) | X | O | 텍스트 생성 |

- **BERT**: 입력 전체를 양방향으로 이해해 문맥을 정밀하게 파악
- **GPT**: 이전 단어만 보면서 다음 단어를 생성
- **T5**: 입력과 출력이 모두 텍스트인 작업을 범용적으로 처리
- **BART**: BERT처럼 인코딩하고 GPT처럼 생성(디코딩)하는 결합형

---

## 실습: Transformer로 감정 분석 (HuggingFace)

사전학습된 Transformer(`distilbert-base-uncased`)를 IMDb 영화 리뷰 감정 분류에 파인튜닝하는 실습.

### 1) 환경 준비

```bash
pip uninstall -y numpy
pip install numpy==1.24.4 --no-cache-dir   # 라이브러리 충돌 방지를 위한 numpy 재설치
pip install -q datasets==2.18.0 transformers==4.41.0 fsspec==2024.2.0 --no-cache-dir
# 설치 후 세션 재시작 필요
```

### 2) 데이터셋 로딩과 토크나이저

```python
import torch
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.optim import Adam
from datasets import load_dataset
from torch.nn.functional import cross_entropy

dataset = load_dataset("imdb")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)

encoded_dataset = dataset.map(tokenize, batched=True)
encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

train_loader = DataLoader(encoded_dataset["train"], shuffle=True, batch_size=8)
test_loader  = DataLoader(encoded_dataset["test"], batch_size=8)
```

- `padding="max_length"`, `max_length=256`: 문장 길이를 256 토큰으로 맞춤 (긴 리뷰는 잘림)

### 3) 모델과 옵티마이저

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased").to(device)
optimizer = Adam(model.parameters(), lr=2e-5)
```

### 4) 학습 / 평가

```python
model.train()
for epoch in range(1):
    total_loss = 0
    for batch in train_loader:
        inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
        labels = batch["label"].to(device)
        outputs = model(**inputs)
        loss = cross_entropy(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"[Epoch {epoch+1}] Loss: {total_loss / len(train_loader):.4f}")

# 평가
model.eval()
correct = 0; total = 0
with torch.no_grad():
    for batch in test_loader:
        ...
        preds = outputs.logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
print(f"[Test Accuracy] {correct / total * 100:.2f}%")
```

### 5) 추론

```python
for i, sample in enumerate(samples):
    text = sample["text"]
    label = "positive" if sample["label"] == 1 else "negative"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    pred_label = "positive" if pred == 1 else "negative"
```

> [!note] 이 실습의 핵심
> Transformer를 밑바닥부터 구현하는 것이 아니라, **사전학습된 Transformer 인코더(BERT 계열)에 분류 헤드를 붙여 파인튜닝**하는 방식.
> 토크나이저가 만든 `input_ids`/`attention_mask`가 곧 임베딩 레이어의 입력이 되고, `DistilBertForSequenceClassification` 안에 인코더 스택 + Linear 분류층이 들어 있음.

참고: https://ig.ft.com/generative-ai/
