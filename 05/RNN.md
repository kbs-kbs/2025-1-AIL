나는 너를 좋아해. 학습시킬 때

1. '나는'의 hidden state 구하기
h₁ = f(W · x₁ + U · h₀)
2. '너를'의 hidden state 구하기
h₂ = f(W · x₂ + U · h₁) 
3. '좋아해'의 hidden state 구하기
h₃ = f(W · x₃ + U · h₂)

W와 U는 한 문장 안에서 공유됨

RNN의 한계
- 기울기 소실 문제
- 장기 의존성 문제

대표적인 RNN
- Vanilla RNN
- LSTM: 장기 기억
- GRU: 계산량이 적고 빠름

```mermaid
flowchart LR
x0(["x0"]) --> A0["A"]
x1(["x1"]) --> A1["A"]
x2(["x2"]) --> A2["A"]
xt(["xt"]) --> A3["A"]
A0 -- h1 --> A1
A1 -- h2 --> A2
A2 -- h3 --> A3
A3 --> ht(["ht"])
```

RNN의 입출력
- 일대다: 이미지 설명
- 다대일: 감정분류, 스팸분류
- 다대다: 품사 태깅, 언어번역, 챗봇
