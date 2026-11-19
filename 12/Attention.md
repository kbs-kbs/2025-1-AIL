# Attention Mechanism

## 1. Seq2Seq

- 시퀀스-투-시퀀스(Sequence-to-Sequence): 입력 시퀀스를 받아 출력 시퀀스로 변환하는 모델
  - 번역: `"I go to school"` → `"나는 학교에 간다"`
  - 응답 생성: `"오늘 날씨 어때?"` → `"맑고, 최고 기온은 20도입니다."`
- 핵심 구조: **Encoder + Decoder**

```
Input Sequence → Encoder → Context Vector → Decoder → Output Sequence
```

| 구성 | 역할 |
|---|---|
| Encoder | 입력 시퀀스를 하나의 벡터(context vector)로 압축 |
| Decoder | 그 벡터를 바탕으로 출력 시퀀스를 생성 |

### Seq2Seq의 한계

- 모든 입력 시퀀스를 **고정된 크기의 벡터 하나**로 압축함
- 디코더는 입력 전체를 직접 보지 못하고, 압축된 벡터만으로 출력을 만들어야 함
- → 문장이 길어질수록 중요한 정보가 희석되고 사라짐

> "나는 어제 친구랑 도서관에서 책을 읽고 나서 카페에 가서 아아를 마시고 왔어" → Seq2Seq: ???

**해결 아이디어**: 출력 시퀀스를 하나씩 만들 때, 입력 시퀀스 중 **어느 단어가 중요한지 직접 살피며** 생성하자 → Attention

---

## 2. Attention

- Attention Mechanism: 현재 시점에서 **중요한 정보에 집중**하는 메커니즘
- 사람이 책을 읽을 때 중요한 단어에 집중하듯, 디코딩 시점마다 중요한 입력 단어를 직접 골라 보면 더 좋은 출력이 나옴
- 디코더가 문장을 생성할 때 압축된 벡터뿐 아니라 **전체 입력 시퀀스**를 보고, 각 단어의 중요도를 계산해 가중치를 줌

### Seq2Seq + Attention

- 기존 Seq2Seq: 인코더 **마지막** 출력 하나만 사용
- Attention: 인코더의 **모든 출력(hidden state)** 을 활용 → 디코더가 중요한 단어를 **동적으로 선택**

```
[입력 문장]  나는 → 사과를 → 먹었다
              ↓       ↓        ↓
            [h₁]    [h₂]     [h₃]      ← 인코더의 단어별 hidden state
              ↓ ↓ ↓
   디코더 시점 t마다 attention score 계산 (h₁, h₂, h₃ 각각과 st의 유사도)
              ↓
   context vector 생성 (h₁, h₂, h₃ 과 attention weight의 가중합)
              ↓
   st + ct → 다음 단어 예측
```

- `st`: 지금까지 디코더가 생성한 문맥 정보를 담고 있는 상태 벡터
- `ct`: 현 시점에서 입력 중 가장 중요한 정보를 담고 있는 context vector

### 계산 방법 (4단계)

1. **Attention Score 계산**: 현재 디코더 hidden state `st` 를 기준으로 인코더의 각 `hi` 와의 유사도(score)를 계산
2. **Attention Weight 계산**: score들을 **softmax** 로 정규화 (전체 합 = 1)
3. **Context Vector 계산**: attention weight와 각 시점의 hidden state의 가중합
4. **출력 생성**: 현재의 context vector `ct` 와 hidden state `st` 를 함께 사용해 다음 단어 예측

| 예시 값 | 벡터 |
|---|---|
| h₁ (나는) | [1, 0, 0] |
| h₂ (사과를) | [0, 1, 1] |
| h₃ (먹었다) | [1, 1, 0] |
| st | [2, 1, 1] |

> [!note] Seq2Seq과의 차이
> Seq2Seq은 벡터 하나만 사용하지만, Attention은 **매 디코딩 시점마다 동적으로 생성되는** context vector를 사용함.
> context vector의 **크기는 고정**되어 있지만, **시점마다 내용이 다름** → 입력 시퀀스의 모든 단어를 부분적으로 반영하면서 중요한 단어에 집중할 수 있음

---

## 3. 실습: 기계 번역 (Attention 적용 Seq2Seq)

### 데이터셋

- `https://www.manythings.org/anki/` 에서 영어-한국어 데이터셋(`kor.txt`) 다운로드
- 형식: `영어문장 \t 한국어문장 \t 출처` (탭으로 구분)

```python
import string

# 데이터 확인
l = []
with open("/content/drive/MyDrive/Colab Notebooks/data/kor.txt", "r", encoding="utf-8") as f:
    lines = f.read().split("\n")
    for line in lines:
        # 특수 문자를 지우고 모든 문자를 소문자로 변환
        txt = "".join(v for v in line if v not in string.punctuation).lower()
        l.append(txt)
```

### 토큰과 BOW(단어 사전)

- `<SOS>`: Start of Sequence — 문장의 시작, **디코딩을 시작**하라는 신호
- `<EOS>`: End of Sequence — 문장의 끝, **디코딩을 끝내라**는 신호

```python
def get_BOW(corpus):
    BOW = {"<SOS>": 0, "<EOS>": 1}
    for line in corpus:
        for word in line.split():
            if word not in BOW.keys():
                BOW[word] = len(BOW.keys())
    return BOW
```

### 데이터셋 클래스

```python
class Eng2Kor(Dataset):
    def __init__(self):
        self.eng_corpus = []   # 영어 문장
        self.kor_corpus = []   # 한국어 문장
        with open(".../kor.txt", "r", encoding="utf-8") as f:
            lines = f.read().split("\n")
            for line in lines:
                if not line.strip():
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    engtxt = "".join(v for v in parts[0] if v not in string.punctuation).lower()
                    kortxt = "".join(v for v in parts[1] if v not in string.punctuation)
                    # 너무 긴 문장은 제외 (10단어 이하만 사용)
                    if len(engtxt.split()) <= 10 and len(kortxt.split()) <= 10:
                        self.eng_corpus.append(engtxt)
                        self.kor_corpus.append(kortxt)
        self.eng_BOW = get_BOW(self.eng_corpus)
        self.kor_BOW = get_BOW(self.kor_corpus)

    def gen_seq(self, line):
        seq = line.split()
        seq.append("<EOS>")      # 문장 끝에 <EOS> 토큰 추가
        return seq

    def __getitem__(self, i):
        data = np.array([self.eng_BOW[txt] for txt in self.gen_seq(self.eng_corpus[i])])
        label = np.array([self.kor_BOW[txt] for txt in self.gen_seq(self.kor_corpus[i])])
        return data, label
```

### 인코더 / 디코더

```python
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout=0., max_length=11):
        # 출력 단어 인덱스를 hidden_size 차원의 임베딩 벡터로 변환
        self.embedding = nn.Embedding(output_size, hidden_size)

        # 어텐션 가중치를 계산하기 위한 MLP
        self.attention = nn.Linear(hidden_size * 2, max_length)

        # context vector + 임베딩을 합친 후 특징 추출용 MLP
        self.context = nn.Linear(hidden_size * 2, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)   # 단어 분류용
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)
```

디코더의 `forward` (핵심):

```python
def forward(self, x, h, encoder_outputs):
    x = self.embedding(x).view(1, 1, -1)
    x = self.dropout(x)

    # 1) 어텐션 가중치 계산: 현재 입력과 이전 hidden state로 score 계산 → softmax
    attn_weights = self.softmax(
        self.attention(torch.cat((x[0], h[0]), -1)), dim=1)     # (1, max_length)

    # 2) 인코더 출력 전체에 어텐션 가중치를 곱해 context vector 생성
    attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                             encoder_outputs.unsqueeze(0))

    # 3) 인코더 각 시점의 중요도와 입력값을 합침 → MLP 통과
    output = torch.cat((x[0], attn_applied[0]), 1)
    output = self.context(output).unsqueeze(0)   # (1, 1, hidden)
    output = self.relu(output)

    # 4) GRU에 넣어 다음 hidden state 계산
    output, hidden = self.gru(output, h)
    output = self.out(output[0])                 # (1, vocab)
    return output
```

### 학습 루프

```python
encoder = Encoder(input_size=len(dataset.eng_BOW), hidden_size=64).to(device)
decoder = Decoder(64, len(dataset.kor_BOW), dropout=0.1).to(device)
encoder_optimizer = Adam(encoder.parameters(), lr=0.0001)
decoder_optimizer = Adam(decoder.parameters(), lr=0.0001)

for epoch in range(1):
    for data, label in tqdm.tqdm(loader, total=len(dataset)):
        # 인코더 진행 후 마지막 hidden state와 모든 시점 출력(encoder_outputs) 저장
        encoder_output, encoder_hidden = encoder(data, encoder_hidden)
        encoder_outputs[ei] = encoder_output
        decoder_input = torch.tensor([[0]]).to(device)   # <SOS>
        decoder_hidden = encoder_hidden                  # 인코더 마지막 hidden을 디코더 초기 상태로
        # 이후 <EOS>가 나올 때까지 한 단어씩 디코딩하며 손실 누적
```

## 정리

- Seq2Seq은 문장을 벡터 하나로 압축하는 구조라 긴 문장에서 정보 손실이 큼
- Attention은 **디코딩 시점마다** 입력의 어느 부분에 집중할지 가중치를 계산해 context vector를 새로 만듦
- → Transformer(다음 주차)는 이 Attention을 recurrence 없이 Self-Attention으로 확장한 모델

참고: https://wikidocs.net/24996
